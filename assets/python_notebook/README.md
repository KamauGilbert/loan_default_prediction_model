**.ipynb Notebook**

This subfolder contains the Colab notebook detailing the process of creating the final model.

Steps include:

1. Data Preparation and EDA  (✓ Done)
2. Data Preprocessing  (✓ Done)
3. Model Training and evaluation  (✓ Done)
4. Hyperparameter tuning  (❌)
5. Model Inference  (❌)



Here's your revised README section with added emojis for visual appeal:

---

### Metric Evaluation Considerations

In training the model, we will assume the role of the lender and provide the following context about the data and business operations of our hypothetical company, KaGil Lenders:

📊 **Dataset Overview:**
- The dataset consists of 255,347 records collected over the course of three years, representing all the people who have borrowed money from KaGil Lenders since its inception. This implies that the company handles approximately 80,000 to 100,000 borrowers annually on average.
- Lending is the primary source of income for KaGil Lenders.

🎯 **Company Objective:**
- The company's main objective is to reduce the number of loan defaulters.

📈 **Model Performance Focus:**
- Given that the model will likely perform better at identifying the negative class (non-defaulters), we will focus on improving performance for the positive class (defaulters), as this aligns with the company’s priorities.

### Precision, Recall, and F1 Score

🎯 **Precision:** 
- A higher precision means fewer False Positives, i.e., reducing the likelihood of incorrectly classifying a non-defaulter as a defaulter. This ensures that more eligible borrowers receive loans, but it doesn't fully address the issue of defaulters.

⚖️ **Recall:** 
- A higher recall means fewer False Negatives, i.e., reducing the likelihood of incorrectly classifying a defaulter as a non-defaulter. Prioritizing recall might lead to more borrowers being incorrectly classified as defaulters, which could reduce the company's profits since lending is its major income source.

### Company's Decision

🚀 **Company Priority:**
- After thorough discussions, it has been decided that KaGil Lenders prioritizes minimizing loan defaults over maximizing profits. However, there is still a need to be cautious about misclassifying good borrowers as defaulters to some extent.

📈 **Model Strategy:**
- Therefore, in our models, we will prioritize achieving a higher F1 score, which balances precision and recall but with a slight emphasis on recall to align with the company’s goal of minimizing defaulters. We will also take into account the model's accuracy but not as much.

---

Feel free to integrate this into your README!
