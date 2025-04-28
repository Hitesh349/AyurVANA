from tensorflow.keras.applications import MobileNetV2

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow import keras

img_width,img_height=224,224

model=MobileNetV2(weights='imagenet',
                include_top=False,
                input_shape=(img_height,img_width,3)
                )

for (i,layer) in enumerate(model.layers):
    print(f"{i} {layer.__class__.__name__} {layer.trainable}")

for layer in model.layers:
    layer.trainable=False

for (i,layer) in enumerate(model.layers):
    print(f"{i} {layer.__class__.__name__} {layer.trainable}")

def add_layer_at_bottom(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model

"""## Data Prep"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_dir='dataset/train'
val_data_dir='dataset/test'

# from tensorflow.keras.layers.preprocessing.image_preprocessing import HORIZONTAL

train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=45,
                                 width_shift_range=0.3,
                                 height_shift_range=0.3,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

val_datagen=ImageDataGenerator(rescale=1./255)

batch_size=32

train_generator=train_datagen.flow_from_directory(train_data_dir,
                                                  target_size=(img_height,img_width),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

val_generator=val_datagen.flow_from_directory(val_data_dir,
                                              target_size=(img_height,img_width),
                                              batch_size=batch_size,
                                              class_mode='categorical')

train_class_names = set()
num_train_samples=0
for i in train_generator.filenames:
    train_class_names.add(i.split('/')[0])
    num_train_samples+=1
print(num_train_samples)
train_class_names

val_class_names = set()
num_val_samples=0
for i in val_generator.filenames:
    val_class_names.add(i.split('/')[0])
    num_val_samples+=1
print(num_val_samples)
val_class_names

num_classes=len(train_generator.class_indices)
print(num_classes)
FC_head=add_layer_at_bottom(model,
                            num_classes)

main_model=Model(inputs=model.input,
                 outputs=FC_head)

main_model.summary()

"""# Training"""

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("Model.h5",
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=2,
                          verbose=1,
                          restore_best_weights=True)

callbacks=[checkpoint,earlystop]

main_model.compile(loss='categorical_crossentropy',
                   optimizer=RMSprop(learning_rate=0.001),
                   metrics=['accuracy'])

epochs=50

batch_size = 32

history = main_model.fit(train_generator,
                         steps_per_epoch=num_train_samples//batch_size,
                         epochs=epochs,
                         callbacks=callbacks,
                         validation_data=val_generator,
                         validation_steps=num_val_samples//batch_size)

import matplotlib.pyplot as plt

#accuracy
plt.plot(history.history['accuracy'], label= 'train acc')
plt.plot(history.history['val_accuracy'], label= 'val acc')
plt.legend()
# plt.saveig('vcc-acc-rps-1.png')
plt.show

#loss
plt.plot(history.history['loss'], label= 'train loss')
plt.plot(history.history['val_loss'], label= 'val loss')
plt.legend()
# plt.saveig('vcc-loss-rps-1.png')
plt.show

"""## Inference"""



pip install opencv-python

import cv2
out=['Aloevera',
 'Amaranthus Viridis (Arive - Dantu)',
 'Amruthabali',
 'Arali',
 'Castor',
 'Mango',
 'Mint',
 'Neem',
 'Sandalwood',
 'Turmeric'
 ]
img=cv2.imread("test6.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.resize(img,(224,224))
img=img/255.
import matplotlib.pyplot as plt
plt.imshow(img)
img=img.reshape(1,224,224,3)
import numpy as np
res=main_model.predict(img)
print(res)
out[np.argmax(res)]

#predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

y_pred = model.predict(val_generator)

test_y = val_generator.classes

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

tflite_model

with open('model.tflite', 'wb') as model_:
    model_.write(tflite_model)

train_data_dir='dataset/train'
val_data_dir='dataset/test'

plt.figure(figsize=(20,20))
img_folder = r'dataset/train'
for i in range(20):
  file = random.choice(os.listdir(img_folder))
  directory_name = file
  data = r'dataset/train/'+file
  file = random.choice(os.listdir(data))
  image_path = os.path.join(data, file)
  img = mpimg.imread(image_path)
  ax = plt.subplot(4,5,i+1)
  ax.title.set_text(directory_name)
  plt.imshow(img)
  plt.savefig('train-dataset.png')

plt.figure(figsize=(20,20))
img_folder = r'dataset/test'
for i in range(20):
  file = random.choice(os.listdir(img_folder))
  directory_name = file
  data = r'dataset/test/'+file
  file = random.choice(os.listdir(data))
  image_path = os.path.join(data, file)
  img = mpimg.imread(image_path)
  ax = plt.subplot(4,5,i+1)
  ax.title.set_text(directory_name)
  plt.imshow(img)
  plt.savefig('test-dataset.png')

from PIL import Image
from numpy import asarray
out=['Aloevera',
 'Amaranthus Viridis (Arive - Dantu)',
 'Amruthabali',
 'Arali',
 'Castor',
 'Mango',
 'Mint',
 'Neem',
 'Sandalwood',
 'Turmeric'
 ]

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(20,20))
img_folder = r'dataset/test'
for i in range(20):
  file = random.choice(os.listdir(img_folder))
  directory_name = file
  data = r'dataset/test/'+file
  file = random.choice(os.listdir(data))
  image_path = os.path.join(data, file)
  img = Image.open(image_path)
  img = img.resize((224,224))
  img = asarray(img)

  img=img/255.
  img_=img.reshape(1,224,224,3)

  res=model_new.predict(img_)
  print(res)

  img = mpimg.imread(image_path)
  ax = plt.subplot(4,5,i+1)
  ax.title.set_text(out[np.argmax(res)])
  plt.imshow(img)
  plt.savefig('predictions.png')



"""### TRANING BASE MODELS

## Training DenseNet without weighted losses
"""

# Install necessary libraries (if not already installed)
!pip install torch torchvision
!pip install plotly

# Import necessary libraries
import pandas as pd
import cv2
from tqdm.auto import tqdm
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import os

img_path = "Data/images/"
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

"""### Model traning Standard Loss"""

train_data

# Constructing the img_path using the base path, image_id, and .jpg extension
train_data['img_path'] = train_data['image_id'].apply(lambda x: f"data/images/{x}.jpg")
test_data['img_path'] = test_data['image_id'].apply(lambda x: f"data/images/{x}.jpg")
# Check if the first image path in the DataFrame exists
print(os.path.exists(train_data['img_path'][0]))
print(os.path.exists(test_data['img_path'][0]))

initial_distribution = train_data[['healthy', 'multiple_diseases', 'rust', 'scab']].sum()
initial_distribution.plot(kind='bar')
plt.title('Initial Class Distribution')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.show()

labels = train_data[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax(axis=1)

X_train, X_Val, y_train, y_val = train_test_split(
    train_data['image_id'],  # or df.drop(['label'], axis=1) for features only
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


X_Val, X_test, y_val, y_test = train_test_split(
    X_Val,
    y_val,
    test_size=0.27,  # Split the validation set in half
    random_state=42,
    stratify=y_val
)

print(f"Train set size: {len(X_train)}")
print(f"Validation set size: {len(X_Val)}")
print(f"Test set size: {len(X_test)}")

train_distribution = y_train.value_counts(normalize=True)
print("Training Set Distribution:\n", train_distribution)

# Check the distribution in the test set
validation_distribution = y_val.value_counts(normalize=True)
print("Test Set Distribution:\n", validation_distribution)

test_distribution = y_test.value_counts(normalize=True)
print("Test Set Distribution:\n", test_distribution)

# Plot the distributions
fig, ax = plt.subplots(1, 3, figsize=(12, 6))

train_distribution.plot(kind='bar', ax=ax[0], title='Training Set Class Distribution')
validation_distribution.plot(kind='bar', ax=ax[1], title='Validation Set Class Distribution')
test_distribution.plot(kind='bar', ax=ax[2], title='Test Set Class Distribution')



plt.show()

# Define the custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, df, image_ids, labels, transform=None):
        self.df = df
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids.iloc[idx]
        img_path = self.df.loc[self.df['image_id'] == img_id, 'img_path'].values[0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels.iloc[idx]
        return image, torch.tensor(label, dtype=torch.long)

train_transformations = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize all images to 224x224
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomVerticalFlip(),  # Randomly flip images vertically
    transforms.RandomRotation(15),  # Randomly rotate images within a specified degree range
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Randomly jitter brightness, contrast, and saturation
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Randomly apply Gaussian blur
    transforms.ToTensor(),  # Convert images to tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

valid_transformations = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize all images to 224x224 for validation
    transforms.ToTensor(),  # Convert images to tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

test_transformations = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the images to match the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DataLoader setup
# Mapping string labels to integers
label_mapping = {'healthy': 0, 'multiple_diseases': 1, 'rust': 2, 'scab': 3}
numerical_y_train = y_train.map(label_mapping)
numerical_y_val = y_val.map(label_mapping)
numerical_y_test = y_test.map(label_mapping)

batch_size = 8

# Creating dataset instances
train_dataset = CustomImageDataset(train_data, X_train, numerical_y_train, transform=train_transformations)
valid_dataset = CustomImageDataset(train_data, X_Val, numerical_y_val, transform=valid_transformations)
test_dataset = CustomImageDataset(train_data, X_test, numerical_y_test, transform=test_transformations)

# Creating data loaders
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_transformed_images(dataloader, num_images=8):
    images, labels = next(iter(dataloader))
    # Create a grid of images
    grid = make_grid(images[:num_images], nrow=4)  # Adjust nrow according to your preference
    plt.figure(figsize=(12, 8))
    plt.imshow(grid.permute(1, 2, 0))  # because images are [C, H, W]
    plt.title('Example of Transformed Images')
    plt.axis('off')
    plt.show()
    print("Corresponding labels:", labels[:num_images])

show_transformed_images(train_loader)

def analyze_class_distribution(dataloader):
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for images, labels in dataloader:
        for label in labels:
            label_counts[label.item()] += 1
        break  # Only process the first batch for demonstration

    # Visualizing the distribution
    labels, counts = zip(*label_counts.items())
    plt.bar(labels, counts, tick_label=[k for k in label_mapping.keys()])
    plt.title('Class Distribution in One Batch')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

analyze_class_distribution(train_loader)

print("lenght of the train_loader",len(train_loader))
print("lenght of the valid_loader",len(valid_loader))
print("lenght of the test_loader",len(test_loader))

# Set device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Install necessary libraries (if not already installed)
!pip install torch torchvision
!pip install plotly

# Import necessary libraries
import pandas as pd
import cv2
from tqdm.auto import tqdm
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim


# Define the model
class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=4):  # Adjust num_classes as per your dataset
        super(CustomDenseNet, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CustomDenseNet(num_classes=4).to(device)


# Define the validation function
def validate_model(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, -1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / total
    return avg_loss, accuracy


# Define the training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    print(f"Starting training for {num_epochs} epochs.")
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_accuracy': []
    }
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}', leave=True)

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {average_loss}")

        valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, device)
        history['train_loss'].append(average_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_accuracy)
        print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

    print("Training completed.")
    return history

# Main script
if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    model.to(device)
    # history = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, device=device)

import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot both training and validation loss on the same plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss', color='blue')
    plt.plot(epochs, history['valid_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Plot validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['valid_accuracy'], label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.show()

# Call the plotting function with the history
plot_training_history(history)

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_predictions))

    return all_predictions, all_labels

# Assuming model, valid_loader and device are already defined
# all_prediction, all_lable = evaluate_model(model, test_loader, device)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(actuals, predictions, classes):
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Assuming you have actuals and predictions arrays available from the evaluation
plot_confusion_matrix(all_lable, all_prediction, classes=['Healthy', 'Multiple Diseases', 'Rust', 'Scab'])

"""###

### Saving the model
"""

torch.save(model.state_dict(), 'model_state_dict_v2.pth')

"""### Loading the model and Predicitng on the Actually test set"""

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm  # Import tqdm for progress bar

class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=4):  # Adjust num_classes as per your dataset
        super(CustomDenseNet, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CustomDenseNet(num_classes=4).to(device)

# Load the model checkpoint
# if torch.backends.mps.is_available() == "True":
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

device = "mps"

print(f"Using device: {device}")

model = CustomDenseNet(num_classes=4).to(device)
print(model)
model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()

"""### Evaluating the Test Data"""

# Assuming `test_data` from a DataFrame that includes paths and potential 'image_id'
class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define test transformations with the correct resizing if it's 448x448
test_transformations = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize all images to 224x224 for validation
    transforms.ToTensor(),  # Convert images to tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load test images with corrected paths
test_dataset = TestDataset(test_data['img_path'].values, transform=test_transformations)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Rest of the code remains largely unchanged

# Prediction function
def predict(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Predicting", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            predictions.extend(probs.cpu().numpy())  # Move to CPU and convert to NumPy for easier handling
    return predictions

# Run predictions with progress tracking
predictions = predict(model, test_loader, device)

# Create a DataFrame for submission
sub = pd.DataFrame(predictions, columns=['healthy', 'multiple_diseases', 'rust', 'scab'])
sub['image_id'] = test_data['image_id']  # Assuming 'image_id' column exists in test_data
sub = sub[['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab']]  # Arrange columns as required
print(sub.head())

"""## Traning DenseNet with weighted Loss"""

from sklearn.utils.class_weight import compute_class_weight

# labels should be a 1D array of all label indices for the dataset
labels = train_data[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax(axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

# Convert class weights to a tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(class_weights_tensor)

# Define the training function
def train_model_weighted_loss(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    print(f"Starting training for {num_epochs} epochs.")
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_accuracy': []
    }
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}', leave=True)

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {average_loss}")

        valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, device)
        history['train_loss'].append(average_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_accuracy)
        print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

    print("Training completed.")
    return history

# Main script
if __name__ == '__main__':
    model = CustomDenseNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    model.to(device)
    history = train_model_weighted_loss(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, device=device)

plot_training_history(history)

all_prediction, all_lable = evaluate_model(model, test_loader, device)

plot_confusion_matrix(all_lable, all_prediction, classes=['Healthy', 'Multiple Diseases', 'Rust', 'Scab'])

"""### Model Performance Comparison: Before and After Weighted Loss Implementation

#### Metrics After Implementing Weighted Loss:
- **Precision**: 0.8602
- **Recall**: 0.8623
- **F1 Score**: 0.8601

**Classification Report (After Weighted Loss):**
| Class              | Precision | Recall | F1 Score | Support |
|--------------------|-----------|--------|----------|---------|
| Healthy            | 0.90      | 1.00   | 0.95     | 28      |
| Multiple Diseases  | 0.60      | 0.60   | 0.60     | 5       |
| Rust               | 1.00      | 0.91   | 0.95     | 34      |
| Scab               | 0.94      | 0.94   | 0.94     | 32      |
| **Accuracy**       |           |        | 0.93     | 99      |
| **Macro Avg**      | 0.86      | 0.86   | 0.86     | 99      |
| **Weighted Avg**   | 0.93      | 0.93   | 0.93     | 99      |

#### Metrics Before Implementing Weighted Loss:
- **Precision**: 0.8130
- **Recall**: 0.7607
- **F1 Score**: 0.7703

**Classification Report (Before Weighted Loss):**
| Class              | Precision | Recall | F1 Score | Support |
|--------------------|-----------|--------|----------|---------|
| Healthy            | 0.90      | 0.96   | 0.93     | 28      |
| Multiple Diseases  | 0.50      | 0.20   | 0.29     | 5       |
| Rust               | 0.97      | 0.94   | 0.96     | 34      |
| Scab               | 0.88      | 0.94   | 0.91     | 32      |
| **Accuracy**       |           |        | 0.91     | 99      |
| **Macro Avg**      | 0.81      | 0.76   | 0.77     | 99      |
| **Weighted Avg**   | 0.90      | 0.91   | 0.90     | 99      |

### Analysis
Implementing weighted loss significantly improved the precision, recall, and F1 score for the 'Multiple Diseases' class, elevating overall model accuracy and balancing performance across classes. The macro and weighted averages show a clear uplift, highlighting the effectiveness of addressing class imbalance in training.
"""

torch.save(model.state_dict(), 'model/model_state_dict_v3.pth')

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm  # Import tqdm for progress bar

class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=4):  # Adjust num_classes as per your dataset
        super(CustomDenseNet, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CustomDenseNet(num_classes=4).to(device)

# Load the model checkpoint
# if torch.backends.mps.is_available() == "True":
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

device = "mps"

print(f"Using device: {device}")

model = CustomDenseNet(num_classes=4).to(device)
print(model)
model.load_state_dict(torch.load('model/model_state_dict_v3.pth'))
model.eval()

# Assuming `test_data` from a DataFrame that includes paths and potential 'image_id'
class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define test transformations with the correct resizing if it's 448x448
test_transformations = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize all images to 224x224 for validation
    transforms.ToTensor(),  # Convert images to tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load test images with corrected paths
test_dataset = TestDataset(test_data['img_path'].values, transform=test_transformations)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Rest of the code remains largely unchanged

# Prediction function
def predict(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Predicting", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            predictions.extend(probs.cpu().numpy())  # Move to CPU and convert to NumPy for easier handling
    return predictions

# Run predictions with progress tracking
predictions = predict(model, test_loader, device)

# Create a DataFrame for submission
sub = pd.DataFrame(predictions, columns=['healthy', 'multiple_diseases', 'rust', 'scab'])
sub['image_id'] = test_data['image_id']  # Assuming 'image_id' column exists in test_data
sub = sub[['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab']]  # Arrange columns as required
print(sub.head())

"""## Train EfficientNetB1 with Weighted Loss:"""

def custom_efficientnetB1(num_classes, pretrained=True):
    model = models.efficientnet_b1(pretrained=pretrained)
    num_features = model.classifier[1].in_features  # Get the input feature count of the classifier
    model.classifier[1] = nn.Linear(num_features, num_classes)  # Replace the classifier with a new one
    return model

def validate_model(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, -1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / total
    return avg_loss, accuracy

!pip install torch torchvision plotly
!pip install seaborn

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import torch.nn as nn
import torch  # Importing the torch module
import torch.optim as optim
from tqdm.auto import tqdm  # Importing tqdm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os


def custom_efficientnetB1(num_classes, pretrained=True):
    model = models.efficientnet_b1(pretrained=pretrained)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model


def validate_model(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, -1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model_weighted_loss(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    print(f"Starting training for {num_epochs} epochs.")
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_accuracy': []
    }
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}', leave=True)

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {average_loss}")

        valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, device)
        scheduler.step(valid_loss)
        history['train_loss'].append(average_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_accuracy)
        print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

    print("Training completed.")
    return history


def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss', color='blue')
    plt.plot(epochs, history['valid_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['valid_accuracy'], label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.show()


def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_predictions))

    return all_predictions, all_labels


def plot_confusion_matrix(actuals, predictions, classes):
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


# --- Data loading and preprocessing ---
img_path = "Data/images/"
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

train_data['img_path'] = train_data['image_id'].apply(lambda x: f"data/images/{x}.jpg")
test_data['img_path'] = test_data['image_id'].apply(lambda x: f"data/images/{x}.jpg")


labels = train_data[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax(axis=1)

X_train, X_Val, y_train, y_val = train_test_split(
    train_data['image_id'],
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

X_Val, X_test, y_val, y_test = train_test_split(
    X_Val,
    y_val,
    test_size=0.27,
    random_state=42,
    stratify=y_val
)


class CustomImageDataset(Dataset):
    def __init__(self, df, image_ids, labels, transform=None):
        self.df = df
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids.iloc[idx]
        img_path = self.df.loc[self.df['image_id'] == img_id, 'img_path'].values[0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels.iloc[idx]
        return image, torch.tensor(label, dtype=torch.long)


train_transformations = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transformations = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transformations = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_mapping = {'healthy': 0, 'multiple_diseases': 1, 'rust': 2, 'scab': 3}
numerical_y_train = y_train.map(label_mapping)
numerical_y_val = y_val.map(label_mapping)
numerical_y_test = y_test.map(label_mapping)

batch_size = 8

train_dataset = CustomImageDataset(train_data, X_train, numerical_y_train, transform=train_transformations)
valid_dataset = CustomImageDataset(train_data, X_Val, numerical_y_val, transform=valid_transformations)
test_dataset = CustomImageDataset(train_data, X_test, numerical_y_test, transform=test_transformations)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)


# --- Main execution ---
if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    EfficientNetmodelB1 = custom_efficientnetB1(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(EfficientNetmodelB1.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    EfficientNetmodelB1.to(device)
    print("device", device)
    history = train_model_weighted_loss(EfficientNetmodelB1, train_loader, valid_loader, criterion, optimizer,
                                        num_epochs=10, device=device)
    torch.save(EfficientNetmodelB1.state_dict(), 'model/model_state_dict_WEIGHTED_EFFICENTNET.pth')

    # --- Evaluation ---
    all_prediction, all_lable = evaluate_model(EfficientNetmodelB1, test_loader, device)
    plot_confusion_matrix(all_lable, all_prediction, classes=['Healthy', 'Multiple Diseases', 'Rust', 'Scab'])

    plot_training_history(history)  # Plot training history after training

import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot both training and validation loss on the same plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss', color='blue')
    plt.plot(epochs, history['valid_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Plot validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['valid_accuracy'], label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.show()

# Call the plotting function with the history
plot_training_history(history)

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_predictions))

    return all_predictions, all_labels

# Assuming model, valid_loader and device are already defined
# all_prediction, all_lable = evaluate_model(model, test_loader, device)
all_prediction, all_lable = evaluate_model(EfficientNetmodel, test_loader, device)

plot_confusion_matrix(all_lable, all_prediction, classes=['Healthy', 'Multiple Diseases', 'Rust', 'Scab'])

"""## DenseNet121 vs EfficientNetB1 Performance Comparison

### **DenseNet121 Performance Summary**
- **Overall Metrics:**
  - **Precision**: 0.8602
  - **Recall**: 0.8623
  - **F1 Score**: 0.8601
  - **Accuracy**: 93%

- **Class-wise Metrics:**
  - **Class 0**: Precision = 0.90, Recall = 1.00, F1 = 0.95 (High performance)
  - **Class 1**: Precision = 0.60, Recall = 0.60, F1 = 0.60 (Lowest performance)
  - **Class 2**: Precision = 1.00, Recall = 0.91, F1 = 0.95 (High performance)
  - **Class 3**: Precision = 0.94, Recall = 0.94, F1 = 0.94 (High performance)

### **EfficientNetB1 Performance Summary**
- **Overall Metrics:**
  - **Precision**: 0.8759
  - **Recall**: 0.8759
  - **F1 Score**: 0.8759
  - **Accuracy**: 95%

- **Class-wise Metrics:**
  - **Class 0**: Precision = 0.96, Recall = 0.96, F1 = 0.96 (High performance)
  - **Class 1**: Precision = 0.60, Recall = 0.60, F1 = 0.60 (Consistent with DenseNet, still the lowest)
  - **Class 2**: Precision = 0.97, Recall = 0.97, F1 = 0.97 (Slightly better than DenseNet)
  - **Class 3**: Precision = 0.97, Recall = 0.97, F1 = 0.97 (Slightly better than DenseNet)

### Analysis
- **Accuracy**: EfficientNetB1 outperforms DenseNet121 by 2% in accuracy, suggesting slightly better overall classification ability.
- **Macro Averages**: EfficientNetB1 shows slightly higher macro averages across precision, recall, and F1 score, indicating more balanced performance across classes.
- **Class 1 Performance**: Both models struggle with Class 1, achieving only 60% across precision, recall, and F1. This suggests issues with either class representation or distinctiveness from other classes.
- **High Performance Classes**: Both models perform exceptionally well on Classes 2 and 3, with EfficientNetB1 showing a marginal improvement.

## Train EfficientNetB2 with Weighted Loss:
"""

def custom_efficientnetB2(num_classes, pretrained=True):
    # Load a pre-trained EfficientNet
    model = models.efficientnet_b2(pretrained=pretrained)

    # Replace the classifier
    num_features = model.classifier[1].in_features  # Get the input feature count of the classifier
    model.classifier[1] = nn.Linear(num_features, num_classes)  # Replace the classifier with a new one

    return model

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Define the training function
def train_model_weighted_loss(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    print(f"Starting training for {num_epochs} epochs.")
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_accuracy': []
    }
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}', leave=True)

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {average_loss}")



        valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, device)
        scheduler.step(valid_loss)  # This ensures the scheduler sees the validation loss
        history['train_loss'].append(average_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_accuracy)
        print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

    print("Training completed.")
    return history

# Main script
if __name__ == '__main__':
    EfficientNetmodelB2 = custom_efficientnetB2(num_classes=4,pretrained = True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(EfficientNetmodelB2.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    EfficientNetmodelB2.to(device)
    print("device",device)
    history = train_model_weighted_loss(EfficientNetmodelB2, train_loader, valid_loader, criterion, optimizer, num_epochs=10, device=device)
    torch.save(EfficientNetmodelB2.state_dict(), 'model/model_state_dict_WEIGHTED_EFFICENTNETB2.pth')

def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_predictions))

    return all_predictions, all_labels

evaluate_model(EfficientNetmodelB2, test_loader, device)

plot_confusion_matrix(all_lable, all_prediction, classes=['Healthy', 'Multiple Diseases', 'Rust', 'Scab'])

"""### Stacking Ensemble"""

EfficientNetmodelB1 = custom_efficientnetB1(num_classes=4).to(device)
# print(EfficientNetmodel)
EfficientNetmodelB1.load_state_dict(torch.load('model/model_state_dict_WEIGHTED_EFFICENTNETB1.pth'))
EfficientNetmodelB1.eval()

# Define the model
class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=4):  # Adjust num_classes as per your dataset
        super(CustomDenseNet, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the model checkpoint
denseNetModel  = CustomDenseNet(num_classes=4).to(device)
denseNetModel.load_state_dict(torch.load('model/model_state_dict_DenseNet_Weighted_loss_V2.pth'))
denseNetModel.eval()

EfficientNetmodelB2 = custom_efficientnetB2(num_classes=4).to(device)
EfficientNetmodelB2.load_state_dict(torch.load('model/model_state_dict_WEIGHTED_EFFICENTNETB2.pth'))
EfficientNetmodelB2.eval()

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def predict_with_averge_ensemble(models, dataloader):
    all_predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            # Collect predictions from all models
            outputs = [model(inputs) for model in models]
            # Majority voting
            outputs = torch.stack(outputs)  # Shape (num_models, batch_size, num_classes)
            outputs = torch.mean(outputs, dim=0)  # Averaging predictions
            _, predicted = torch.max(outputs, 1)  # Convert probabilities to class predictions
            all_predictions.extend(predicted.cpu().numpy())
    return all_predictions




models = [denseNetModel, EfficientNetmodelB1,EfficientNetmodelB2]
ensemble_predictions = predict_with_averge_ensemble(models, test_loader)

true_label = [label for _, label in test_loader]
true_label = [item.item() for tensor in true_label for item in tensor]
# print(true_label)
# print(ensemble_predictions)

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def evaluate_score(all_labels, all_predictions):
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_predictions))

    return all_predictions, all_labels

all_predictions,all_labels = evaluate_score(true_label, ensemble_predictions)

plot_confusion_matrix(all_lable, all_prediction, classes=['Healthy', 'Multiple Diseases', 'Rust', 'Scab'])

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def predict_with_soft_ensemble(models, dataloader):
    all_predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            # Collect probability predictions from all models
            probabilities = [torch.softmax(model(inputs), dim=1) for model in models]
            # Average the probabilities
            mean_probabilities = torch.mean(torch.stack(probabilities), dim=0)
            # Pick the class with the highest mean probability
            _, predicted = torch.max(mean_probabilities, 1)
            all_predictions.extend(predicted.cpu().numpy())
    return all_predictions

# Assuming models are in evaluation mode and properly output logits for softmax
models = [denseNetModel, EfficientNetmodelB1, EfficientNetmodelB2]  # Your model list here
ensemble_predictions = predict_with_soft_ensemble(models, test_loader)
all_predictions,all_labels = evaluate_score(true_label, ensemble_predictions)

# Continue with any further analysis or evaluation with ensemble_predictions

plot_confusion_matrix(all_lable, all_prediction, classes=['Healthy', 'Multiple Diseases', 'Rust', 'Scab'])

"""# Model Performance Comparison and Analysis

## DenseNet121 (After Implementing Weighted Loss):
- **Precision:** 0.8602
- **Recall:** 0.8623
- **F1 Score:** 0.8601
- **Accuracy:** 93%
- **Detailed Performance:**
  - Significant improvement in handling the 'Multiple Diseases' category compared to results before implementing weighted loss.
  - Balanced performance across most classes, with room for improvement in minor classes.

## EfficientNetB1 (Weighted Loss):
- **Precision:** 0.8759
- **Recall:** 0.8759
- **F1 Score:** 0.8759
- **Accuracy:** 95%
- **Detailed Performance:**
  - Consistently high performance across all categories, with particular strength in handling 'Rust' and 'Scab'.
  - The model performs slightly better than DenseNet121 individually, indicating EfficientNetB1's suitability for this specific dataset.

## Averaging Ensemble: (Averaging Ensemble  of DenseNet121 and EfficientNet Variants):
- **Precision:** 0.9000
- **Recall:** 0.9275
- **F1 Score:** 0.9117
- **Accuracy:** 96%
- **Detailed Performance:**
  - The ensemble method enhances overall accuracy and class-specific metrics.
  - Notably, it improves precision and recall for 'Multiple Diseases', addressing previous imbalances in class performance.

## Soft Voting Ensemble (DenseNet121 and EfficientNet Variants)
- **Precision:** 0.9333
- **Recall:** 0.9348
- **F1 Score:** 0.9337
- **Accuracy:** 97%
- **Detailed Performance:**
  - Soft voting ensemble further improves metrics across all categories, achieving the highest accuracy and macro-average metrics.
  - Particularly strong precision and recall are observed for the 'Multiple Diseases' category, demonstrating enhanced class balance and robustness.

---

## Analysis
- **Impact of Weighted Loss:** Applying weighted loss to both DenseNet121 and EfficientNetB1 models significantly improved class balance, particularly for the 'Multiple Diseases' category, which initially exhibited lower performance.
- **Individual Model Comparison:** EfficientNetB1 slightly outperforms DenseNet121 in almost all metrics, making it a preferable model choice for this dataset when considering individual models.
- **Effectiveness of Ensemble Techniques:** Both the initial voting ensemble and the refined soft voting ensemble showed significant improvement in performance metrics. The soft voting ensemble achieved the highest overall metrics, demonstrating its effectiveness in leveraging the strengths of both models for balanced and accurate predictions.

## Conclusion
The application of weighted loss, combined with ensemble techniques, led to substantial improvements in model performance, particularly in class balance for challenging categories like 'Multiple Diseases'. EfficientNetB1, with weighted loss, emerged as the best individual model, while the soft voting ensemble of DenseNet121 and EfficientNet variants achieved the highest overall accuracy and balanced metrics.

This analysis suggests that:
1. **Ensemble methods**, particularly soft voting, are highly effective in combining model strengths and enhancing overall performance.
2. **Weighted loss** is beneficial for class-imbalanced datasets, improving recall and precision for underrepresented classes.
3. Future improvements could explore **more advanced ensemble methods** like stacking, as well as targeted **data augmentation** for minor classes.

The soft voting ensemble method provides the most robust solution, achieving the highest accuracy and consistent performance across all classes, making it the ideal approach for tasks that require high accuracy and class balance.

## Grad-CAM visualizations

### Visualization for base models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision import transforms
from torchvision.utils import make_grid
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

models = [denseNetModel, EfficientNetmodelB1, EfficientNetmodelB2]
model_names = ['CustomDenseNet121', 'CustomEfficientNetB1', 'CustomEfficientNetB2']

class SafeGradCAM(GradCAM):
    def __del__(self):
        try:
            self.activations_and_grads.release()
        except AttributeError:
            pass  # Ignore if the attribute does not exist

# Function to apply Grad-CAM and get model prediction
def apply_gradcam(model, target_layer, input_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    cam = SafeGradCAM(model=model, target_layers=[target_layer])
    target_category = None  # Assuming the highest scoring category
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None

    with torch.no_grad():  # Ensure to do a forward pass without gradients accumulation
        outputs = model(input_tensor.unsqueeze(0))
    predicted = outputs.argmax(dim=1).item()

    grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Take the CAM for the first image in the batch
    return grayscale_cam, predicted

# Sample images randomly from the test_loader
random_indices = random.sample(range(len(test_loader.dataset)), 10)
selected_images = [test_loader.dataset[i][0] for i in random_indices]
labels = ["Healthy", "Multiple Diseases", "Rust", "Scab"]
selected_labels = [labels[test_loader.dataset[i][1]] for i in random_indices]  # Fetch the true labels

# Setup visualization
fig, axs = plt.subplots(len(selected_images), len(models) + 1, figsize=(15, 4 * len(selected_images)))  # +1 for true label

for i, image_tensor in enumerate(selected_images):
    image_numpy = image_tensor.numpy().transpose((1, 2, 0))
    image_numpy = (image_numpy - image_numpy.min()) / (image_numpy.max() - image_numpy.min())  # Normalize to 0-1 range

    # Display true label
    axs[i, 0].imshow(image_numpy)
    axs[i, 0].axis('off')
    axs[i, 0].set_title(f'True Label: {selected_labels[i]}')

    for j, model in enumerate(models):
        if model_names[j] == 'CustomDenseNet121':
            target_layer = model.features.norm5  # Last BN layer in DenseNet
        elif model_names[j] == 'CustomEfficientNetB1':
            target_layer = model.features[-1]  # Last block in EfficientNetB1
        elif model_names[j] == 'CustomEfficientNetB2':
            target_layer = model.features[-1]  # Last block in EfficientNetB2

        grayscale_cam, prediction = apply_gradcam(model, target_layer, image_tensor)

        # Visualize heatmap on image
        heatmap = show_cam_on_image(image_numpy, grayscale_cam, use_rgb=True)
        axs[i, j+1].imshow(heatmap)
        axs[i, j+1].axis('off')
        axs[i, j+1].set_title(f'{model_names[j]}\nPred: {labels[prediction]}')

plt.tight_layout()
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# #TRAINING AND PREDICTION
# %%capture
# !pip install graphviz

import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image, display

data1=pd.read_csv("Crop_Production.csv")

X = data1[['Year']]
y = data1['Value']

models = {
    'Decision Tree': DecisionTreeRegressor(),
    'Linear Regression': LinearRegression(),
    'XGBoost': XGBRegressor(),
    'Random Forest': RandomForestRegressor()
}

predictions = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions[model_name] = model.predict(X_test)

evaluation = {}
for model_name, y_pred in predictions.items():
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    evaluation[model_name] = {'MSE': mse, 'MAE': mae}

# Visualize the decision tree model using Graphviz
dot_data = export_graphviz(models['Decision Tree'], out_file=None,
                           feature_names=X.columns, filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)

fig = go.Figure()
for model_name, y_pred in predictions.items():
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_pred,
        mode='markers',
        name=model_name
    ))
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_test,
    mode='markers',
    name='Actual'
))
fig.update_layout(
    title='Healthy Crop Yield Prediction',
    xaxis_title='Data Point Index',
    yaxis_title='Pesticides Use (tonnes of active ingredients)',
)
fig.show()

print('Evaluation Results:')
for model_name, metrics in evaluation.items():
    print(f'{model_name}:')
    print(f'MSE: {metrics["MSE"]}')
    print(f'MAE: {metrics["MAE"]}')
    print('---')

# Create a Graphviz object from the dot file
graph = graphviz.Source(dot_data)

# Set the format and filename to save the image
image_format = 'png'
image_filename = 'graph_Healthy_Crop_Yield_Production'

# Save the Graphviz visualization as an image
graph.format = image_format
graph.render(filename=image_filename, format=image_format, cleanup=True)

# Display the image with a specific size
image_path = f'{image_filename}.{image_format}'
display(Image(filename=image_path, width=1000, height=800))

# Display the graph (large version - might not render well in all environments)
display(graph)
