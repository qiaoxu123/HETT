from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import from_bounds, transform as window_transform


def _pixel(contour: list[list[float]], transform) -> np.ndarray:
    points = []
    for x, y in contour:
        col, row = (~transform) * (float(x), float(y))
        points.append([round(col), round(row)])
    return np.asarray(points, dtype=np.int32)


@lru_cache(maxsize=2)
def _read_rgb(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_template(rgb: np.ndarray, raster_transform, target: dict, anchors: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    h, w = rgb.shape[:2]
    masks = {name: np.zeros((h, w), np.uint8) for name in ("target", "anchor")}
    for item, name in [(target, "target"), *[(a, "anchor") for a in anchors]]:
        if item.get("contour"):
            poly = _pixel(item["contour"], raster_transform)
            poly[:, 0] -= int(target["crop_col"])
            poly[:, 1] -= int(target["crop_row"])
            cv2.fillPoly(masks[name], [poly], 255)
    return rgb, masks


def build_episode_templates(data_root: Path, out_root: Path, row: dict, sizes=(40, 80, 120), image_size=384) -> list[dict]:
    """Build true north-up orthophoto crops; these are top-down templates only."""
    map_name = row["map_name"]
    objects = json.loads((data_root / "cityrefer" / "objects.json").read_text())[map_name]
    target = objects[str(row["target_object_id"])]
    anchors = [o for o in objects.values() if o.get("name", "").casefold() in {x.casefold() for x in row.get("landmark_names", [])}]
    xy = row["target_xy"]
    image_path = data_root / "rgbd" / f"{map_name}.png"
    records = []
    rgb = _read_rgb(str(image_path))
    with rasterio.open(data_root / "rgbd" / f"{map_name}.tif") as ds:
        for metres in sizes:
            half = metres / 2
            window = from_bounds(xy[0]-half, xy[1]-half, xy[0]+half, xy[1]+half, transform=ds.transform)
            window = window.round_offsets().round_lengths()
            col0,row0=int(window.col_off),int(window.row_off); width,height=int(window.width),int(window.height)
            crop = np.full((height,width,3),127,np.uint8)
            sx0,sy0=max(0,col0),max(0,row0); sx1,sy1=min(rgb.shape[1],col0+width),min(rgb.shape[0],row0+height)
            if sx1>sx0 and sy1>sy0: crop[sy0-row0:sy1-row0,sx0-col0:sx1-col0]=rgb[sy0:sy1,sx0:sx1]
            t = window_transform(window, ds.transform)
            crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
            masks = {"target": np.zeros((image_size,image_size),np.uint8), "anchor": np.zeros((image_size,image_size),np.uint8)}
            for item, key in [(target,"target"), *[(a,"anchor") for a in anchors]]:
                contour = item.get("contour")
                if not contour: continue
                pts=[]
                for x,y in contour:
                    col,rowpix=(~t)*(float(x),float(y))
                    pts.append([round(col*image_size/max(1,width)),round(rowpix*image_size/max(1,height))])
                cv2.fillPoly(masks[key],[np.asarray(pts,np.int32)],255)
            dest = out_root / "templates" / row["split"] / row["episode_id"] / f"topdown_{metres}m.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            mask_dir = dest.parent / f"topdown_{metres}m_masks.npz"
            np.savez_compressed(mask_dir, **masks)
            records.append({"episode_id":row["episode_id"],"split":row["split"],"map_name":map_name,
                "target_object_id":row["target_object_id"],"target_xy":xy,"scene_size_m":metres,
                "view":"top_down","image":str(dest.relative_to(out_root)),"masks":str(mask_dir.relative_to(out_root)),
                "north_up_orthophoto":True,"oblique_available":False,"fpv_available":False})
    return records
