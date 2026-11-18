"""Build candidate-level examples directly from PathSnap raw trip files.

The current label rule maps each GPS observation to the nearest arc in the
supplied route sequence, constrained to non-decreasing route order. It is an
explicit working assumption (not a supplied point-to-arc mapping); inspect and
report its limitations before treating labels as authoritative.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

EARTH_M = 6_371_008.8


def read_numeric(path):
    # NumPy's parser avoids creating millions of short Python lists for .nodes/.arcs.
    data=np.loadtxt(path,dtype=float)
    return np.atleast_2d(data)


def project_many(lon_lat, lon0, lat0):
    scale_x=EARTH_M*math.cos(math.radians(lat0))*math.pi/180
    scale_y=EARTH_M*math.pi/180
    out=np.empty((len(lon_lat),2),dtype=np.float64)
    out[:,0]=(lon_lat[:,0]-lon0)*scale_x
    out[:,1]=(lon_lat[:,1]-lat0)*scale_y
    return out


def point_segment(point, start, end):
    # Scalar arithmetic is much faster than allocating several NumPy arrays for
    # each of the millions of point/segment distance checks in candidate search.
    dx=float(end[0]-start[0]); dy=float(end[1]-start[1])
    px=float(point[0]-start[0]); py=float(point[1]-start[1])
    denom=dx*dx+dy*dy
    frac=0.0 if denom==0 else max(0.0,min(1.0,(px*dx+py*dy)/denom))
    ex=px-frac*dx; ey=py-frac*dy
    return math.hypot(ex,ey),frac


def bearing(a,b):
    dx,dy=b-a
    return math.atan2(dx,dy)


class SegmentGrid:
    """Grid index over segment bounding boxes (no buffer expansion)."""
    def __init__(self, starts, ends, cell=100.0, buffer=800.0, indices=None):
        self.starts,self.ends=starts,ends
        self.cell,self.buffer=cell,buffer
        self.cells=defaultdict(list)
        for i in (range(len(starts)) if indices is None else indices):
            a,b=starts[i],ends[i]
            ix0=math.floor(min(a[0],b[0])/cell); ix1=math.floor(max(a[0],b[0])/cell)
            iy0=math.floor(min(a[1],b[1])/cell); iy1=math.floor(max(a[1],b[1])/cell)
            for ix in range(ix0,ix1+1):
                for iy in range(iy0,iy1+1): self.cells[(ix,iy)].append(i)

    def nearby(self, point, radius):
        c=self.cell; ix0=math.floor((point[0]-radius)/c); ix1=math.floor((point[0]+radius)/c)
        iy0=math.floor((point[1]-radius)/c); iy1=math.floor((point[1]+radius)/c)
        ids=set()
        for ix in range(ix0,ix1+1):
            for iy in range(iy0,iy1+1): ids.update(self.cells.get((ix,iy),()))
        scored=[]
        for i in ids:
            d,f=point_segment(point,self.starts[i],self.ends[i])
            if d<=radius: scored.append((d,i,f))
        scored.sort()
        return scored


def process_trip(folder: Path, k=8, radius=100.0, max_radius=800.0):
    tid=folder.name
    track=read_numeric(folder/f"{tid}.track")
    nodes_ll=read_numeric(folder/f"{tid}.nodes")
    arcs=read_numeric(folder/f"{tid}.arcs").astype(int)
    route=read_numeric(folder/f"{tid}.route").astype(int).ravel()
    if len(track)==0 or len(arcs)==0 or len(nodes_ll)==0 or len(route)==0:
        return [], {"trip_id":tid,"observations":len(track),"candidate_misses":len(track)}
    lon0=float(np.mean(track[:,0])); lat0=float(np.mean(track[:,1]))
    nodes=project_many(nodes_ll,lon0,lat0)
    starts=nodes[arcs[:,0]]; ends=nodes[arcs[:,1]]
    grid=SegmentGrid(starts,ends,buffer=max_radius)
    points=project_many(track[:,:2],lon0,lat0)
    ts=track[:,2]
    # Progress-constrained nearest route arc; route position never moves backward.
    route_valid=route[(route>=0)&(route<len(arcs))]
    route_arcs=np.unique(route_valid)
    route_grid=SegmentGrid(starts,ends,buffer=max_radius,indices=route_arcs)
    route_positions=defaultdict(list)
    for j,arc_id in enumerate(route_valid): route_positions[int(arc_id)].append(j)
    ref_ids=[]; cursor=0
    for point in points:
        best=None
        for search_radius in (radius,max_radius):
            for d,arc_id,_ in route_grid.nearby(point,search_radius):
                for j in route_positions.get(arc_id,()):
                    if j>=cursor and (best is None or d<best[0]): best=(d,j,arc_id)
            if best is not None: break
        # GPS outliers can land beyond the route index; use the full remaining
        # route only as a fallback, keeping the common case spatially indexed.
        if best is None:
            for j in range(cursor,len(route_valid)):
                arc_id=int(route_valid[j]); d,_=point_segment(point,starts[arc_id],ends[arc_id])
                if best is None or d<best[0]: best=(d,j,arc_id)
        if best is None:
            ref_ids.append(-1)
        else:
            cursor=best[1]; ref_ids.append(best[2])
    degrees=np.bincount(arcs.ravel(),minlength=len(nodes))
    headings=[]; speeds=[]; dts=[]
    for i,p in enumerate(points):
        if i==0: headings.append(0.0);speeds.append(0.0);dts.append(0.0)
        else:
            dt=max(float(ts[i]-ts[i-1]),1e-3); delta=points[i]-points[i-1]
            headings.append(bearing(points[i-1],p));speeds.append(float(np.linalg.norm(delta)/dt));dts.append(dt)
    output=[]; misses=0
    for i,p in enumerate(points):
        found=[]; r=radius
        while r<=max_radius and not found:
            found=grid.nearby(p,r); r*=2
        found=found[:k]
        if not found: misses+=1; continue
        true_arc=ref_ids[i]
        if true_arc not in {int(c[1]) for c in found}: misses+=1
        dmin=found[0][0]
        for rank,(dist,arc_id,frac) in enumerate(found):
            a,b=arcs[arc_id]; seg_bearing=bearing(starts[arc_id],ends[arc_id])
            hd=0.0 if i==0 else math.atan2(math.sin(headings[i]-seg_bearing),math.cos(headings[i]-seg_bearing))
            output.append({"trip_id":tid,"observation_id":str(i),"arc_id":str(arc_id),
                "distance_m":dist,"from_node_distance_m":float(np.linalg.norm(p-starts[arc_id])),
                "to_node_distance_m":float(np.linalg.norm(p-ends[arc_id])),"fraction_along_arc":frac,
                "arc_length_m":float(np.linalg.norm(ends[arc_id]-starts[arc_id])),
                "heading_cos":math.cos(hd),"heading_abs_diff_rad":abs(hd),"gps_speed_mps":speeds[i],
                "time_gap_s":dts[i],"candidate_rank":rank,"distance_gap_m":dist-dmin,
                "candidate_count":len(found),"from_node_degree":int(degrees[a]),
                "to_node_degree":int(degrees[b]),"is_match":int(arc_id==true_arc and true_arc>=0)})
    return output,{"trip_id":tid,"observations":len(points),"candidate_misses":misses}


def load_partition(root: Path, **kwargs):
    """Return in-memory rows and trip summaries for train or student_test root."""
    rows=[]; summaries=[]
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        trip_rows,summary=process_trip(folder,**kwargs)
        rows.extend(trip_rows);summaries.append(summary)
    return rows,summaries
