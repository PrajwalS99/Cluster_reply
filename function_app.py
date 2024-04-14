import azure.functions as func
import logging
import json
from Dummy_Model import SemiSupervisedClassifier, extract_text_from_blob, preprocess

dummy_model = SemiSupervisedClassifier(topics)

@app.blob_trigger(name="myblob", path="clusterreplycontainer",
                  connection="clusterreplyblob_STORAGE") 
def blob_trigger_cluster_reply(myblob: func.InputStream):
    try:
        # Extract text from the blob
        extracted_text = extract_text_from_blob(myblob.read())

        # Preprocess the extracted text
        preprocessed_text = preprocess(extracted_text)

        # Classify the preprocessed text using the dummy model
        probabilities = dummy_model.predict_proba([preprocessed_text])[0]

        # Convert probabilities to JSON
        result = json.dumps({topic: prob for topic, prob in zip(topics, probabilities)})

        # Log the result
        logging.info(f"Classified probabilities: {result}")

        # Return the classified probabilities
        return func.HttpResponse(result, mimetype="application/json")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return func.HttpResponse("An error occurred", status_code=500)
