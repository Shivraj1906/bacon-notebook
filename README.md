### **Flexible Paleoclimate Age-Depth Models Using an Autoregressive Gamma Process**  
An interactive Jupyter Notebook for demonstrating the Bacon age-depth model.  

## **Dependencies**  
This notebook is tested on **Python 3.10** and **Python 3.12**. It requires `pip` to install dependencies. All modules used are cross-platform.  

### **Required Python Packages**  
Install the following dependencies before running the notebook:  

- `numpy`  
- `scipy`  
- `matplotlib`  
- `pandas`  
- `numba` (for optimization)  

## **Setup Instructions**  
To install the required dependencies, run the following command in your terminal:  

```sh
pip install -r requirements.txt
```  

Once installed, simply execute the notebook.  

## **Performance Notes**  
- The MCMC run takes approximately **10-15 minutes** on a **2.6 GHz processor**.  
- Operations are vectorized where possible to improve efficiency.  
- **Calibration Curve Optimization:**  
  - The calibration curve data is **precomputed** using linear interpolation to save time.  
- **Numba Acceleration:**  
  - The `G` function is compiled to machine code using **Numba**, significantly improving performance.  

## **Usage**  
- Load your dataset following the required format.  
- Run the notebook to generate the age-depth model.  
- Adjust parameters as needed to explore different model behaviors.  