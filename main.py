import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from classifiers.xcm import xcm
from classifiers.xcm_seq import xcm_seq
from classifiers.mtex_cnn import mtex_cnn
from utils.utils import load_dataset


if __name__ == "__main__":
    
    # Parameters
    dataset = sys.argv[1]
    model_name = sys.argv[2]
    batch_size = int(sys.argv[3])
    if model_name in ['XCM', 'XCM-Seq']:
        window_size = float(sys.argv[4])
    else:
        window_size = 0
    epochs = 100
    n_splits = 5
    
    # Load dataset
    x_train, y_train, x_test, y_test, y_train_nonencoded, y_test_nonencoded = load_dataset(dataset)
    
    # Instantiate the cross validator
    skf = StratifiedKFold(n_splits=n_splits, random_state=123, shuffle=True)
    
    # Instantiate the results dataframe
    results = pd.DataFrame(columns=['Dataset','Model_Name','Batch_Size','Window_Size','Fold','Accuracy_Train','Accuracy_Validation','Accuracy_Test'])
    
    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(skf.split(x_train, y_train_nonencoded)):
        print('Training on fold ' + str(index+1))
    
        # Generate batches from indices
        xtrain, xval = x_train[train_indices], x_train[val_indices]
        ytrain, yval, ytrain_nonencoded, yval_nonencoded = y_train[train_indices], y_train[val_indices], y_train_nonencoded[train_indices], y_train_nonencoded[val_indices]

        # Train the model
        model_dict = {'XCM': xcm, 'XCM-Seq': xcm_seq, 'MTEX-CNN': mtex_cnn}
        if model_name in ['XCM', 'XCM-Seq']:
            model = model_dict[model_name](input_shape=xtrain.shape[1:], n_class=ytrain.shape[1], window_size=window_size)
        else:
            model = model_dict[model_name](input_shape=xtrain.shape[1:], n_class=ytrain.shape[1])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(xval, yval))
        
        # Calculate accuracies 
        acc_train = accuracy_score(ytrain_nonencoded, np.argmax(model.predict(xtrain), axis=1))
        acc_val = accuracy_score(yval_nonencoded, np.argmax(model.predict(xval), axis=1))
        acc_test = accuracy_score(y_test_nonencoded, np.argmax(model.predict(x_test), axis=1))
        
        # Add fold results to the dedicated dataframe
        results.loc[index] = [dataset, model_name, batch_size, int(window_size*100), index+1, acc_train, acc_val, acc_test]
        
    # Train the model on the full train set
    print('Training on the full train set')
    model_dict = {'XCM': xcm, 'XCM-Seq': xcm_seq, 'MTEX-CNN': mtex_cnn}
    if model_name in ['XCM', 'XCM-Seq']:
        model = model_dict[model_name](input_shape = x_train.shape[1:], n_class=y_train.shape[1], window_size=window_size)
    else:
        model = model_dict[model_name](input_shape=x_train.shape[1:], n_class=y_train.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose = 1)
    
    # Add result to the results dataframe
    results['Accuracy_Test_Full_Train'] = accuracy_score(y_test_nonencoded, np.argmax(model.predict(x_test), axis=1))
        
    # Save results to CSV
    results.to_csv('./results/results'+'_'+dataset+'_'+model_name+'_'+str(batch_size)+'_'+str(int(window_size*100))+'.csv', index=False)
    print(results)