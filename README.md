# Fraud-detection-using-pyspark-udf
Group Project for CSE-598 Data-Intensive Systems for Machine Learning


# Fraud Detection via Spark UDFs and Custom CNN

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Environment Setup](#environment-setup)  
4. [Methodology](#methodology)  
5. [Results](#results)  
6. [Skills & Lessons Learned](#skills--lessons-learned)  
7. [Future Work](#future-work)  
8. [Authors](#authors)  
9. [References](#references)  
10. [License](#license)  

---

## Project Overview
Traditional identity-verification pipelines rely on external services or manual review, introducing latency and security risks. This project embeds a lightweight custom CNN into Apache Spark via PySpark UDFs, enabling real-time, in-database fraud detection on ID images without moving data outside the cluster.

---

## Dataset
The simulated identity-verification dataset consists of three CSV files in the `data/` directory:
- **idmeta.csv**: User metadata (ID, timestamp, document type)  
- **idlabel.csv**: Ground-truth labels (`0` = genuine, `1` = fraudulent)  
- **idimage.csv**: Base64-encoded image strings  

---

## Environment Setup
We evaluated and compared four platforms for in-database inference:
- **Velox**: Complex build, limited documentation → abandoned  
- **PostgreSQL UDFs**: No native tensor support → limited to metadata queries  
- **EvaDB**: No multi-dimensional tensor/CNN support → unsuitable for images  
- **Apache Spark**: Distributed processing + Python UDFs → chosen for scalability and ease of integration  

---

## Methodology
1. **Model Architecture**  
   - Compared a pretrained EfficientNetB0 (81% validation accuracy, prone to overfitting) with a custom-designed CNN optimized for our ID-fraud domain.  
   - Custom CNN uses three convolutional layers, max-pooling, and fully-connected layers for binary classification.
   - ![image](https://github.com/user-attachments/assets/3a4df484-e9a0-4227-bebd-8a8435fc9382)
   - ![image](https://github.com/user-attachments/assets/1dbf4732-3abf-4630-915b-fad375fc28dd)



2. **Spark Integration**  
   - Export trained PyTorch model to disk.  
   - Implement a Python function to decode Base64 images, preprocess, and perform inference.  
   - Register that function as a PySpark UDF and invoke it directly within Spark SQL queries.
   - ![image](https://github.com/user-attachments/assets/86fe281c-0f6d-4c5b-bfe9-581a33d6aef3)


---

## Results
![image](https://github.com/user-attachments/assets/e9ee4741-43e7-4fc4-aa49-6d490ebfd8d0)
![image](https://github.com/user-attachments/assets/6e054521-07bd-44f4-9104-00070b42dd38)
![image](https://github.com/user-attachments/assets/2728d017-061d-4f14-b885-80c0dbf67755)
![image](https://github.com/user-attachments/assets/f0d4ea9d-043d-4162-8f0d-e2a1acb2a164)

- **Accuracy**  
  - EfficientNetB0: ~81% but unstable on subtle fraud patterns  
  - Custom CNN: Comparable accuracy, faster training, better generalization  
- **Performance**  
  - Spark UDFs scale linearly across cores/nodes  
  - Outperform single-machine Pandas pipelines at medium and large data volumes  

---

## Skills & Lessons Learned
- Evaluating platform trade-offs (Velox, PostgreSQL, EvaDB, Spark)  
- Designing lightweight CNN architectures for domain-specific image tasks  
- Bridging Python and JVM via PySpark UDFs for in-database ML  
- Benchmarking distributed inference performance and identifying bottlenecks  

---

## Future Work
- Train on diverse, global ID formats for better generalization  
- Fuse image data with metadata and geolocation for ensemble scoring  
- Extend UDF approach to other engines (DuckDB, PostgresPL/Python)  
- Support near-real-time inference via Spark Structured Streaming  
- Add explainability (e.g., Grad-CAM) to highlight model decisions  

---

## Authors
- Himansh U. — hmudigon@asu.edu  
- Jainil T. — jtrived7@asu.edu  
- Bala Sujith P. — bpotinen@asu.edu  
- Vishrut S. — vshah80@asu.edu  

_All authors are students in the School of Computing and Augmented Intelligence, Arizona State University._

---

## References
1. Spark UDF documentation  
2. PyTorch model deployment guides  
3. EvaDB and PostgreSQL UDF literature  
4. EfficientNet research paper  

---

## License
This project is licensed under the MIT License.  
```
