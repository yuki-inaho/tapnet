# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Live Demo for Online TAPIR."""

import sys
import argparse
import functools
import time

import cv2
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Optional

from pathlib import Path

# @HACK: This is a hack to import tapnet module.
#  File "/home/inaho-omen/Project/tapnet/live_demo.py", line 28, in <module>
#    from tapnet import tapir_model
# ModuleNotFoundError: No module named 'tapnet'
current_dir_path = str(Path(__file__).parent)
parent_dir_path_of_current_dir = str(Path(current_dir_path).parent.absolute())
print(f"The python search path is added: {parent_dir_path_of_current_dir}")
sys.path.append(parent_dir_path_of_current_dir)

from tapnet import tapir_model


NUM_POINTS = 8
RESIZED_HEIGHT = 256
MAX_TRY_COUNT_TO_CAPTURE_FRAMES = 5
DUMP_FPS = 30


def construct_initial_causal_state(num_points, num_resolutions):
    """Construct initial causal state."""
    value_shapes = {
        "tapir/~/pips_mlp_mixer/block_1_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_1_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_2_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_2_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_3_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_3_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_4_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_4_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_5_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_5_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_6_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_6_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_7_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_7_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_8_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_8_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_9_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_9_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_10_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_10_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_11_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_11_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_causal_2": (1, num_points, 2, 2048),
    }
    fake_ret = {k: jnp.zeros(v, dtype=jnp.float32) for k, v in value_shapes.items()}
    return [fake_ret] * num_resolutions * 4


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.astype(np.float32)
    frames = frames / 255 * 2 - 1
    return frames


def postprocess_frames(frames):
    """Postprocess frames back to traditional image format.

    Args:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32

    Returns:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8
    """
    frames = (frames + 1) / 2 * 255
    frames = np.round(frames).astype(np.uint8)
    return frames


def postprocess_occlusions(occlusions, exp_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
      occlusions: [num_points, num_frames], [-inf, inf], np.float32
      exp_dist: [num_points, num_frames], [-inf, inf], np.float32

    Returns:
      visibles: [num_points, num_frames], bool
    """
    # visibles = occlusions < 0
    pred_occ = jax.nn.sigmoid(occlusions)
    pred_occ = 1 - (1 - pred_occ) * (1 - jax.nn.sigmoid(exp_dist))
    return pred_occ < 0.5


def load_checkpoint(checkpoint_path):
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    return ckpt_state["params"], ckpt_state["state"]


def build_online_model_init(frames, points):
    model = tapir_model.TAPIR(use_causal_conv=True)
    feature_grids = model.get_feature_grids(frames, is_training=False)
    features = model.get_query_features(
        frames,
        is_training=False,
        query_points=points,
        feature_grids=feature_grids,
    )
    return features


def build_online_model_predict(frames, features, causal_context):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR(use_causal_conv=True)
    feature_grids = model.get_feature_grids(frames, is_training=False)
    trajectories = model.estimate_trajectories(
        frames.shape[-3:-1],
        is_training=False,
        feature_grids=feature_grids,
        query_features=features,
        query_points_in_video=None,
        query_chunk_size=64,
        causal_context=causal_context,
        get_causal_context=True,
    )
    causal_context = trajectories["causal_context"]
    del trajectories["causal_context"]
    return {k: v[-1] for k, v in trajectories.items()}, causal_context


def get_frame(video_capture):
    r_val, image = video_capture.read()
    trunc = np.abs(image.shape[1] - image.shape[0]) // 2
    if image.shape[1] > image.shape[0]:
        image = image[:, trunc:-trunc]
    elif image.shape[1] < image.shape[0]:
        image = image[trunc:-trunc]
    return r_val, image


def main(capture_input: Union[str, int], dump_video_flag: bool):
    print("Welcome to the TAPIR live demo.")
    print("Please note that if the framerate is low (<~12 fps), TAPIR performance")
    print("may degrade and you may need a more powerful GPU.")

    print("Loading checkpoint...")
    # --------------------
    # Load checkpoint and initialize
    params, state = load_checkpoint(str(Path(current_dir_path, "checkpoints/causal_tapir_checkpoint.npy")))

    print("Creating model...")
    online_init = hk.transform_with_state(build_online_model_init)
    online_init_apply = jax.jit(online_init.apply)

    online_predict = hk.transform_with_state(build_online_model_predict)
    online_predict_apply = jax.jit(online_predict.apply)

    rng = jax.random.PRNGKey(42)
    online_init_apply = functools.partial(online_init_apply, params=params, state=state, rng=rng)
    online_predict_apply = functools.partial(online_predict_apply, params=params, state=state, rng=rng)

    print("Initializing the VideoCapture module...")
    vc = cv2.VideoCapture(capture_input)
    try_count_capturing = 0
    if vc.isOpened():  # try to get the first frame
        while True:
            rval, frame = get_frame(vc)
            if not rval:
                try_count_capturing += 1
                if try_count_capturing >= MAX_TRY_COUNT_TO_CAPTURE_FRAMES:
                    print("Unable to capture frames from the camera: try_count_capturing >= MAX_TRY_COUNT_TO_CAPTURE_FRAMES")
                    sys.exit(1)
            else:
                break
    else:
        raise ValueError("Unable to open camera.")

    image_height, image_width, _ = frame.shape
    resized_rate = RESIZED_HEIGHT / image_height
    resize_image = functools.partial(cv2.resize, dsize=None, fx=resized_rate, fy=resized_rate)
    image_height_resized, image_width_resized, _ = resize_image(frame).shape
    print(f"The size of truncated input images: {image_width}x{image_height}")
    print(f"The size of target images: {image_width_resized}x{image_height_resized}")

    # --------------------
    # Start point tracking
    pos = tuple()
    query_frame = True
    have_point = [False] * NUM_POINTS
    query_features = None
    causal_state = None
    next_query_idx = 0

    print("Compiling jax functions (this may take a while...)")
    # --------------------
    # Call one time to compile
    query_points = jnp.zeros([NUM_POINTS, 3], dtype=jnp.float32)
    query_features, _ = online_init_apply(
        frames=preprocess_frames(frame[None, None]),
        points=query_points[None, 0:1],
    )
    jax.block_until_ready(query_features)

    query_features, _ = online_init_apply(
        frames=preprocess_frames(frame[None, None]),
        points=query_points[None],
    )
    causal_state = construct_initial_causal_state(NUM_POINTS, len(query_features.resolutions) - 1)
    (prediction, causal_state), _ = online_predict_apply(
        frames=preprocess_frames(frame[None, None]),
        features=query_features,
        causal_context=causal_state,
    )

    jax.block_until_ready(prediction["tracks"])

    last_click_time = 0

    def mouse_click(event, x, y, flags, param):
        del flags, param
        global pos, query_frame, last_click_time

        # event fires multiple times per click sometimes??
        if (time.time() - last_click_time) < 0.5:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            pos = (y, frame.shape[1] - x)
            query_frame = True
            last_click_time = time.time()

    # Set up the video writer
    writer: Optional[cv2.VideoWriter] = None
    if dump_video_flag:
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        writer = cv2.VideoWriter(str(Path(current_dir_path, "output", "output.mp4")), fourcc, DUMP_FPS, (image_width_resized, image_height_resized))

    cv2.namedWindow("Point Tracking")
    cv2.setMouseCallback("Point Tracking", mouse_click)

    t = time.time()
    step_counter = 0

    while rval:
        rval, frame = get_frame(vc)
        frame = resize_image(frame)
        if query_frame:
            query_points = jnp.array((0,) + pos, dtype=jnp.float32)

            init_query_features, _ = online_init_apply(
                frames=preprocess_frames(frame[None, None]),
                points=query_points[None, None],
            )
            init_causal_state = construct_initial_causal_state(1, len(query_features.resolutions) - 1)

            # cv2.circle(frame, (pos[0], pos[1]), 5, (255,0,0), -1)
            query_frame = False

            def upd(s1, s2):
                return s1.at[:, next_query_idx : next_query_idx + 1].set(s2)

            causal_state = jax.tree_map(upd, causal_state, init_causal_state)
            query_features = tapir_model.QueryFeatures(
                lowres=jax.tree_map(upd, query_features.lowres, init_query_features.lowres),
                hires=jax.tree_map(upd, query_features.hires, init_query_features.hires),
                resolutions=query_features.resolutions,
            )
            have_point[next_query_idx] = True
            next_query_idx = (next_query_idx + 1) % NUM_POINTS
        if pos:
            (prediction, causal_state), _ = online_predict_apply(
                frames=preprocess_frames(frame[None, None]),
                features=query_features,
                causal_context=causal_state,
            )
            track = prediction["tracks"][0, :, 0]
            occlusion = prediction["occlusion"][0, :, 0]
            expected_dist = prediction["expected_dist"][0, :, 0]
            visibles = postprocess_occlusions(occlusion, expected_dist)
            track = np.round(track)

            for i in range(len(have_point)):
                if visibles[i] and have_point[i]:
                    cv2.circle(frame, (int(track[i, 0]), int(track[i, 1])), 5, (255, 0, 0), -1)
                    if track[i, 0] < 16 and track[i, 1] < 16:
                        print((i, next_query_idx))
        cv2.imshow("Point Tracking", frame[:, ::-1])
        if pos:
            step_counter += 1
            if time.time() - t > 5:
                print(f"{step_counter/(time.time()-t)} frames per second")
                t = time.time()
                step_counter = 0
        else:
            t = time.time()
        if dump_video_flag:
            writer.write(frame[:, ::-1])
        key = cv2.waitKey(1)

        if key == 27:  # exit on ESC
            break
    if dump_video_flag:
        writer.release()

    cv2.destroyWindow("Point Tracking")
    vc.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video or camera feed.")
    parser.add_argument("-i", "--input", type=str, default="0", help="path to the video file or integer for the camera id")
    parser.add_argument("-hs", "--height-of-input-images", type=int, default=240, help="height of input images")
    parser.add_argument("--dump-video", action="store_true", help="Dump processed video to file (default: False)")
    args = parser.parse_args()
    capture_input = args.input if not args.input.isdigit() else int(args.input)
    main(capture_input, args.dump_video)
