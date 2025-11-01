# depthai_camera.py (replace the class with this)

import sys, math
import numpy as np
import depthai as dai
import cv2

class DepthAiCamera(object):
    def __init__(self, width=640, height=400, fps=30, disable_rgb=False, align_to_rgb=True):
        np.set_printoptions(threshold=sys.maxsize)
        self.width  = width
        self.height = height
        self.fps = fps
        self.disable_rgb = disable_rgb
        self.align_to_rgb = bool(align_to_rgb)

        extended_disparity = True
        subpixel = False
        lr_check = True

        pipeline = dai.Pipeline()

        # ---------- RGB ----------
        camRgb = None
        if not self.disable_rgb:
            camRgb = pipeline.create(dai.node.ColorCamera)
            xoutRgb = pipeline.create(dai.node.XLinkOut)
            xoutRgb.setStreamName("rgb")

            camRgb.setPreviewSize(width, height)
            camRgb.setInterleaved(False)
            camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            camRgb.setFps(fps)
            camRgb.preview.link(xoutRgb.input)

        # ---------- Stereo ----------
        monoLeft  = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo    = pipeline.create(dai.node.StereoDepth)
        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setFps(fps); monoRight.setFps(fps)
        monoLeft.setCamera("left"); monoRight.setCamera("right")

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(lr_check)
        stereo.setExtendedDisparity(extended_disparity)
        stereo.setSubpixel(subpixel)
        stereo.setRectifyEdgeFillColor(0)  # black borders for invalids

        # Align depth to RGB if requested
        if self.align_to_rgb and not self.disable_rgb:
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align to RGB camera
            stereo.setOutputSize(width, height)                # make depth exactly match RGB preview size

        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        stereo.depth.link(xoutDepth.input)

        # ---------- Boot device ----------
        self.pipeline = pipeline
        self.device   = dai.Device(self.pipeline)
        self.queDepth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        self.queRgb   = None if self.disable_rgb else self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        # cache calibration and intrinsics
        self._calib = self.device.readCalibration()
        self._K_rgb  = np.array(self._calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height))
        self._K_left = np.array(self._calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, width, height))
        self._K_right= np.array(self._calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, width, height))

    def __exit__(self, *_):
        try: self.device.close()
        except: pass

    def stop(self):
        try: self.device.close()
        except: pass

    @property
    def depth_aligned_to_rgb(self) -> bool:
        # true when we requested align_to_rgb and RGB stream is enabled
        return bool(self.align_to_rgb and not self.disable_rgb)

    def get_intrinsics(self):
        """
        Returns:
          K_rgb, K_depth
        Note: if depth is aligned to RGB, K_depth == K_rgb.
        """
        if self.depth_aligned_to_rgb:
            return self._K_rgb.copy(), self._K_rgb.copy()
        else:
            # Depth is in the rectified RIGHT camera frame by default
            return self._K_rgb.copy(), self._K_right.copy()

    def get_frames(self):
        colorFrame = None
        if self.queRgb is not None:
            inRgb = self.queRgb.tryGet()
            if inRgb is not None:
                colorFrame = inRgb.getCvFrame()

        inDepth = self.queDepth.tryGet()
        depthFrame = inDepth.getFrame() if inDepth is not None else None
        return colorFrame, depthFrame

    def get_images(self):
        colorFrame, depthFrame = self.get_frames()
        if depthFrame is None:
            return colorFrame, None
        depth_m = np.asanyarray(depthFrame).astype(np.float32) * 0.001
        depth_m[depth_m == 0] = math.nan
        return colorFrame, depth_m
