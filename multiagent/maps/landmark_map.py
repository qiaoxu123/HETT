import cv2
import numpy as np
import Levenshtein
import re

from multiagent.cityreferobject import get_landmarks, remove_duplicate_landmarks_by_area

from .map import Map
from typing import List, Dict, Callable, Tuple

class LandmarkMap(Map):
    _landmarks_cache = None
    _landmark_segmentations = None

    def __init__(
        self,
        map_name: str,
        map_shape: Tuple[int, int],
        pixels_per_meter: float,
        landmark_names: List[str],
        min_similarity: float = 0.0,
    ):
        super().__init__(map_name, map_shape, pixels_per_meter)
        matches = LandmarkMap._search_landmarks_by_name(
            map_name, landmark_names, min_similarity=min_similarity
        )
        self.landmark_names = [query for query, _, _ in matches]
        self.landmarks = [landmark for _, landmark, _ in matches]
        self.landmark_confidences = [confidence for _, _, confidence in matches]

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
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', name.casefold())

    @classmethod
    def _search_landmarks_by_name(
        cls,
        map_name: str,
        query_names: List[str],
        min_similarity: float = 0.0,
    ):
        # load landmark data
        if cls._landmarks_cache is None:
            cls._landmarks_cache = remove_duplicate_landmarks_by_area(get_landmarks())

        landmarks = list(cls._landmarks_cache[map_name].values())
        matches = []
        for query in query_names:
            normalized_query = cls._normalize_name(query)
            if not normalized_query or not landmarks:
                continue
            scored = []
            for landmark in landmarks:
                normalized_name = cls._normalize_name(landmark.name)
                edit_distance = Levenshtein.distance(normalized_name, normalized_query)
                denominator = max(len(normalized_name), len(normalized_query), 1)
                scored.append((1.0 - edit_distance / denominator, landmark))
            confidence, landmark = max(scored, key=lambda item: item[0])
            if confidence >= min_similarity:
                matches.append((query, landmark, confidence))
        return matches
