# few-shot-object-retrieval


##### dependecies
 - keras==2.0.8
 - tensorflow-gpu==1.14
 - skimage
 - This code assumes you have the latest xmlpathology reposititory in your pythonpath: https://github.com/computationalpathologygroup/xml-pathology/tree/alpha/


##### Create enconding
The following line will create an .npy encoded representation for the gives wsi with the given model. An optional tissue mask can be specified to speed up the encoding.
 - python encode.py --model_path='path_to_encoder_model' --image_path='path_to_wsi' --mask_path='path_to_mask' --output_path='path_to_output'


##### Detect
The following line will create an xml file with detections for the encoded wsis. See fsorparameters.yml for more options.
 - python detect.py --query_encoded_path='path_to_encoding.npy' 


### Acknowledgements

Created in the [#EXAMODE](https://www.examode.eu/) project
