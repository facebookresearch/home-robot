import argparse
import json

import cv2
import numpy as np
from performance_utils import extract_clean_map


def main(args):
    gt_map = extract_clean_map(args.map_gt)
    pred_map = extract_clean_map(args.map_pred)
    window = np.concatenate([gt_map, pred_map], axis=1)
    _, W = gt_map.shape[:2]
    cv2.namedWindow("Correspondences")
    points_gt = []
    points_pred = []

    # mouse callback function
    def get_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(window, (x, y), 2, (255, 0, 0), -1)
            num_points = len(points_gt) + len(points_pred)
            if num_points % 2 == 0:
                assert x < W and x >= 0
                points_gt.append((x, y))
            else:
                assert x >= W and x < 2 * W
                points_pred.append((x - W, y))

    cv2.setMouseCallback("Correspondences", get_point)

    while True:
        cv2.imshow("Correspondences", window)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    save_data = {
        "gt": points_gt,
        "pred": points_pred,
        "gt_path": args.map_gt,
        "pred_path": args.map_pred,
    }
    with open(args.save_path, "w") as fp:
        json.dump(save_data, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_gt", required=True, type=str)
    parser.add_argument("--map_pred", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)

    args = parser.parse_args()

    main(args)
