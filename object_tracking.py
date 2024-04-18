from bytetrack.byte_track import ByteTrack, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

from supervision import ColorPalette
from supervision import Point, VideoInfo, VideoSink, get_video_frames_generator
from supervision import BoxAnnotator,TraceAnnotator, LineZone, LineZoneAnnotator
from supervision import Detections

from typing import List
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import argparse

class ObjectTracking:
    def __init__(self, input_video_path, output_video_path) -> None:
        self.model = YOLO("yolo/yolov8x.pt")
        self.model.fuse()
        
        # dict maping class_id to class_name
        self.CLASS_NAMES_DICT = self.model.model.names
        # class_ids of interest - car, motocycle, bus and truck
        self.CLASS_ID = [2, 3, 5, 7]

        self.LINE_START = Point(50, 1500)
        self.LINE_END = Point(3840, 1500)
        
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        
        # create BYTETracker instance
        self.byte_tracker = ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
        # create VideoInfo instance
        self.video_info = VideoInfo.from_video_path(self.input_video_path)
        # create frame generator
        self.generator = get_video_frames_generator(self.input_video_path)
        # create LineZone instance
        self.line_zone = LineZone(start=self.LINE_START, end=self.LINE_END)
        # create instance of BoxAnnotator
        self.box_annotator = BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        # create instance of TraceAnnotator
        self.trace_annotator = TraceAnnotator(thickness=4, trace_length=50)
        # create LineZoneAnnotator instance
        self.line_zone_annotator = LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
    
    def callback(self, frame, index):
        # model prediction on single frame and conversion to supervision Detections
        results = self.model(frame, verbose = False)[0]
        detections = Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, self.CLASS_ID)]
        
        detections = self.byte_tracker.update_with_detections(detections)
        
        labels = [
        f"#{tracker_id} {self.model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
        
        annotated_frame = self.trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
        annotated_frame= self.box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels)

        # update line counter
        self.line_zone.trigger(detections)
        # return frame with box and line annotated result
        return  self.line_zone_annotator.annotate(annotated_frame, line_counter=self.line_zone)
    
    def process(self):
        with VideoSink(target_path=self.output_video_path, video_info=self.video_info) as sink:
            for index, frame in enumerate(
                get_video_frames_generator(source_path=self.input_video_path)
            ):
                result_frame = self.callback(frame, index)
                sink.write_frame(frame=result_frame)