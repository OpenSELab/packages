# packages
Software developers often improve and guarantee software quality through intelligent information and insights. To observe such intelligent insights, researchers and practitioners have widely introduced machine learning classification techniques from various data analytics tools (e.g., Python and R) into software engineering to mine rich software data. However, such intelligent insights should be stable and consistent to help developers improve the software quality. Otherwise, developers would lost trust in software analytics. Unfortunately, researchers and practitioners currently used various data analytics tools indiscriminately and interchangeably, which have mined intelligent insights with no clear consensus. Such lack of clear directive on how apply data analytics tools prevents developers effectively introduce these intelligent insights into practice. Therefore, we first systematically explore the effectiveness of different data ananlytics tools on the machine learning classifiers across performance consistency, performance stability, and model interpretation in software engineering. Through a case study of 23 popular software datasets (from three domains of software engineering: defect prediction datasets, issue lifetime estimation datasets, and code smell detection datasets), we observe that: i) a given classifiers with the tune setting from different machine learning packages would generate inconsistent and unstable insights among different data analytics tools; ii) different classifiers behave differently due to different data analytics packages across the performance consistent, performance stability and model interpretation, even in case of the studied packages with the same setting. Such findings seriously threaten the stability of insights across the replicated studies. Therefore, on the basis of these findings, we suggest that: 1) always expose not only the data analytics tools (e.g., python or R) but also used machine learning packages when they discover the intelligent insights for their tasks based on the machine learning models, especially with the tune setting; 2) When researchers and practitioners aim to build the models with high and stable performance for the software tasks, trying the classifier packages in the R would be a better choice; and 3) When using RF classifier to build the models for software tasks, using the RF classifier packages with the tune setting would be a better choice.

The following four tables are the p-values of the Wilcoxon signed-rank test for RQ1 and RQ2, where * indicates p≤0.05 and # indicates p>0.05. These results correspond to Figure 2 to Figure 5 in the paper.
![](https://github.com/OpenSELab/packages/blob/main/data/wilcoxon/Wilcoxon%20test%20Table%201-github-upload.jpg)
![](https://github.com/OpenSELab/packages/blob/main/data/wilcoxon/Wilcoxon%20test%20Table%202-github-upload.jpg)
![](https://github.com/OpenSELab/packages/blob/main/data/wilcoxon/Wilcoxon%20test%20Table%203-github-upload.jpg)
![](https://github.com/OpenSELab/packages/blob/main/data/wilcoxon/Wilcoxon%20test%20Table%204-github-upload.jpg)
