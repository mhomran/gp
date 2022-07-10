import cv2
def _write_hint(img, msg, coord, color=(0, 0, 0)):
    cv2.putText(img, msg, (int(coord[0][0]), int(coord[1][0])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
