import argparse
import os
import time
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import detect_and_align
import cognitive_face as CF


def predictFace(img, person_group_id=2, large_person_group_id=None, max_candidates_return=2, threshold=0.5):
    faces = CF.face.detect(img)
    res = CF.face.identify([faces[0]['faceId']], person_group_id, large_person_group_id, max_candidates_return, threshold)
    return res


def load_model(model):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Loading model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        raise ValueError('Specify model file, not directory!')


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Setup models
            load_model(args.model)
            mtcnn = detect_and_align.create_mtcnn(sess, None)

            cap = cv2.VideoCapture(0)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            show_landmarks = False
            show_bb = True
            show_fps = True
            while(True):
                start = time.time()
                _, frame = cap.read()

                # Locate faces and landmarks in frame
                face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(frame, mtcnn)

                if len(face_patches) > 0:

                    for bb, landmark in zip(padded_bounding_boxes, landmarks):
                        if show_bb:
                            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
                        if show_landmarks:
                            for j in range(5):
                                size = 1
                                top_left = (int(landmark[j]) - size, int(landmark[j + 5]) - size)
                                bottom_right = (int(landmark[j]) + size, int(landmark[j + 5]) + size)
                                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)
                else:
                    print('Couldn\'t find a face')

                end = time.time()

                seconds = end - start
                fps = round(1 / seconds, 2)

                if show_fps:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(fps), (0, int(frame_height) - 5), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow('frame', frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    show_landmarks = not show_landmarks
                elif key == ord('b'):
                    show_bb = not show_bb
                elif key == ord('f'):
                    show_fps = not show_fps

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to model protobuf (.pb) file')
    main(parser.parse_args())
