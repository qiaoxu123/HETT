import cv2
import numpy as np
import Levenshtein

from multiagent.cityreferobject import get_landmarks, remove_duplicate_landmarks_by_area

from .map import Map
from typing import List, Dict, Callable, Optional, Tuple

class LandmarkMap(Map):
    _landmarks_cache = None
    _landmark_segmentations = None

    def __init__(
        self,
        map_name: str,
        map_shape: Tuple[int, int],
        pixels_per_meter: float,
        landmark_names: Optional[List[str]],
    ):
        super().__init__(map_name, map_shape, pixels_per_meter)
        if landmark_names is None:
            self.landmarks = LandmarkMap._all_landmarks(map_name)
            self.landmark_names = [landmark.name for landmark in self.landmarks]
        else:
            self.landmark_names = landmark_names
            self.landmarks = LandmarkMap._search_landmarks_by_name(map_name, landmark_names)

        self.landmark_map = np.zeros(map_shape, dtype=np.uint8)
        for lm in self.landmarks:
            self.landmark_map = cv2.fillPoly(
                img=self.landmark_map ,
                pts=[np.stack(self.to_rows_cols(lm.contour))[::-1].T],
                color=1
            )

    def get_contours(self):
        contours = []
        for lm in self.landmarks:
            # print(len(lm.contour))
            # contour = []
            # for point in lm.contour:
            #     contour.append(np.array([point.x, point.y]))
            # contour = np.stack(contour)
            contours.append(lm.contour)
        return contours
    
    def to_array(self, dtype=np.float32) -> np.ndarray:
        return self.landmark_map[np.newaxis].astype(dtype)
    
    @classmethod
    def _ensure_landmark_cache(cls):
        if cls._landmarks_cache is None:
            cls._landmarks_cache = remove_duplicate_landmarks_by_area(get_landmarks())

    @classmethod
    def _all_landmarks(cls, map_name: str):
        cls._ensure_landmark_cache()
        return list(cls._landmarks_cache[map_name].values())

    @classmethod
    def _search_landmarks_by_name(cls, map_name: str, query_names: List[str]):
        cls._ensure_landmark_cache()
        landmarks = list(cls._landmarks_cache[map_name].values())
        if not landmarks:
            return []
        return [
            min(landmarks, key=lambda lm, q=query: Levenshtein.distance(lm.name, q))
            for query in query_names
        ]
