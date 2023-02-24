class DataSet:
    
    # Loads a MNIST dataset
    @classmethod
    def load_mnist_dataset(cls,dataset, path):
        # Scan all the directories and create a list of labels
        labels = os.listdir(os.path.join(path, dataset))
        # Create lists for samples and labels
        X = []
        y = []
        # For each label folder
        for label in labels:
            # And for each image in given folder
            for file in os.listdir(os.path.join(path, dataset, label)):
                # Read the image
                image = cv2.imread(
                os.path.join(path, dataset, label, file),
                cv2.IMREAD_UNCHANGED)
                
                # And append it and a label to the lists
                X.append(image)
                y.append(label)

        # Convert the data to proper numpy arrays and return
        return np.array(X), np.array(y).astype('uint8')

    # MNIST dataset (train + test)
    @classmethod
    def create_data_mnist(cls,path):
        # Load both sets separately
        X, y = cls.load_mnist_dataset('train', path)
        X_test, y_test = cls.load_mnist_dataset('test', path)
        # And return all the data
        return X, y, X_test, y_test