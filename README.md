# Calibrate LLM-as-Judge

LLM's are often used by tools such as _TruLens_ to evaluate an application response or it's intermediate steps, such as retrieved context.

We can ensure the validity of these evaluations by comparing the evaluation scores against ground truth evaluations. This process can be referred to as *calibration*.

This app gives you a simple wizard to run calibration experiments to test the performance of different LLM-judges against ground truth evaluation datasets.

1. Run the app with `streamlit run app.py`
2. Choose a dataset
   ![select_dataset](https://github.com/user-attachments/assets/5d1897b4-2298-40c4-acf4-8e26b11110b0)

3. Select the number of samples from the dataset to test
   ![select_samples](https://github.com/user-attachments/assets/ca663e9d-3074-4146-a78a-f5e39e96529c)
   
5. Choose LLM-judges (more feedback functions coming soon)
    ![choose_models](https://github.com/user-attachments/assets/d31a792c-8ab3-4b01-a427-163ce6058c18)

6. Pick metrics
   ![choose_metrics](https://github.com/user-attachments/assets/79e2770d-f1da-4516-a3ee-a81547f2f27b)

8. Chose the k-value (for applicable metrics)
    ![choose_k](https://github.com/user-attachments/assets/b86370ae-ae07-4160-9fe4-b0a95f216a5f)

9. Run experiment!
    ![run_experiment](https://github.com/user-attachments/assets/af3f3bd8-169b-4a3c-93fc-3d9e90fd444a)

10. View results
    ![view_results](https://github.com/user-attachments/assets/c3f203fe-50df-4bc4-ac8d-a19cdd721f84)
