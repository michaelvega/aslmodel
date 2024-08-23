import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier


def getImg(outputF):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    detector = HandDetector(maxHands=1)

    offset = 20
    imgSize = 300

    folder = f"../res/MP_Data/{outputF}"  # MUST CHANGE
    counter = 0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Error: Failed to read frame from camera.")
            raise "Could not read frame from camera."
        hands, img = detector.findHands(img)
        if not hands:
            print("No hands detected.")
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            save_success = cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(f'{folder}/Image_{time.time()}.jpg')
            if save_success:
                print(f'Image saved to {folder}/Image_{time.time()}')
            print(counter)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def testModel(target_letter):
    print("cupcake")
    cap = cv2.VideoCapture(0)


    detector = HandDetector(maxHands=1)
    classifier = Classifier("../lib/modelalpha-2/keras_model.h5", "../lib/modelalpha-2/labels.txt")  # MUST CHANGE

    offset = 20
    imgSize = 300

    labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
              "v", "w", "x", "y"]

    while cap.isOpened():
        success, img = cap.read()

        imgOutput = img.copy()

        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if str(labels[index]) == target_letter:
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (0, 255, 0), 4)
            else:
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def testSingleFrame():
    detector = HandDetector(maxHands=1)
    classifier = Classifier("../lib/modelalpha-2/keras_model.h5", "../lib/modelalpha-2/labels.txt")

    offset = 20
    imgSize = 300

    labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
              "v", "w", "x", "y"]

    ###

    img = cv2.imread("../res/test/in/test-b.png")  # CHANGE

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

    cv2.imwrite('../res/test/out/testOutput-b1.png', imgOutput)
    cv2.imwrite('../res/test/out/testOutput-b2.png', imgCrop)
    cv2.imwrite('../res/test/out/testOutput-b3.png', imgWhite)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #target_letter = str(input("Input the Letter you want to practice: "))
    #testModel(target_letter)
    # testSingleFrame()
    folder = str(input("Word for input: ")).strip()
    getImg(folder)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
