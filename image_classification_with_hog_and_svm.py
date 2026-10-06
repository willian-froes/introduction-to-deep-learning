from pathlib import Path
import cv2
import numpy as np

positive_path = Path("assets/hand_dataset/positives")
negative_path = Path("assets/hand_dataset/negatives")

positive_dataset = list(positive_path.glob("*.png"))
negative_dataset = list(negative_path.glob("*.png"))

dataset = positive_dataset + negative_dataset
labels = np.array([1] * len(positive_dataset) + [0] * len(negative_dataset))
images = [cv2.imread(image) for image in dataset]

grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
resized_images = [cv2.resize(image, (256, 256)) for image in grayscale_images]

hog = cv2.HOGDescriptor(
    _winSize=(256, 256),
    _blockSize=(16, 16), 
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

features = [hog.compute(image) for image in resized_images]
X = np.array(features)

features_indexes = np.arange(len(X))

positive_indexes = features_indexes[:len(positive_dataset)]
negative_indexes = features_indexes[len(positive_dataset):]

np.random.seed(42)

np.random.shuffle(positive_indexes)
np.random.shuffle(negative_indexes)

train_indexes = np.concatenate([positive_indexes[:21], negative_indexes[:21]])
test_indexes = np.concatenate([positive_indexes[21:], negative_indexes[21:]])

X_train = X[train_indexes]
X_test = X[test_indexes]

y_train = labels[train_indexes]
y_test = labels[test_indexes]

svm = cv2.ml.SVM_create()

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(1.0)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

_, test_predictions = svm.predict(X_test)
_, train_predictions = svm.predict(X_train)

test_predictions = test_predictions.ravel().astype(np.int32)
train_predictions = train_predictions.ravel().astype(np.int32)

test_accuracy = np.mean(test_predictions == y_test)
train_accuracy = np.mean(train_predictions == y_train)

qualitative_examples = {
    "VERDADEIRO POSITIVO": None,
    "VERDADEIRO NEGATIVO": None,
    "FALSO POSITIVO": None,
    "FALSO NEGATIVO": None
}

print("\nClassificação das imagens do conjunto de teste: ")

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

print("\nÍndice | Real | Predito | Resultado")
print("-------------------------------------")

for index, (real, predicted) in enumerate(zip(y_test, test_predictions)):
    result = f"{GREEN}ACERTO{RESET}" if real == predicted else f"{RED}ERRO{RESET}"
    qualitative_result = None

    if real == 1 and predicted == 1:
        qualitative_result = "VERDADEIRO POSITIVO"
    elif real == 0 and predicted == 0:
        qualitative_result = "VERDADEIRO NEGATIVO"
    elif real == 0 and predicted == 1:
        qualitative_result = "FALSO POSITIVO"
    else:
        qualitative_result = "FALSO NEGATIVO"

    if qualitative_examples[qualitative_result] is None:
        qualitative_examples[qualitative_result] = index

    print(f"{index:6} | {real:4} | {predicted:7} | {result}")

print(f"\nPrecisão dos conjuntos: Treino = {train_accuracy * 100:.2f}%, Teste = {test_accuracy * 100:.2f}%")

examples = []

for result, index in qualitative_examples.items():
    if index is None: continue

    image = images[test_indexes[index]]

    height, width = image.shape[:2]
    size = min(height, width)

    image = image[(height - size) // 2:(height + size) // 2, (width - size) // 2:(width + size) // 2]
    image = cv2.resize(image, (512, 512))

    cv2.putText(image, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    examples.append(image)

qualitative_examples_image = cv2.vconcat([cv2.hconcat(examples[:2]), cv2.hconcat(examples[2:])])

cv2.imshow("Exemplos qualitativos", qualitative_examples_image)
cv2.waitKey(0)
cv2.destroyAllWindows()