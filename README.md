# AISC 2007 - Case Study 2
This is the AISC 2007 - Deep Learning Case Study 2 assignment performed on AI and Data Science (M07 Group 2) course performed by Group 2.
- Instructor: Bhavik Gandhi
- Project Demonstration: 
- Website Deployed: https://cuddly-chainsaw-645xx5qjjx9fxxr4-8501.app.github.dev/
The website is deployed through Streamlit sharing. However as model.h5 files have heavy size over 100 MB, .h5 files couldn’t be pushed into this github repository.

# Models in .h5:
1. Transfer learnt model from a model trained on EfficientNet/VGG by adding additional layers: https://drive.google.com/file/d/1ne9HHLhYYKN_R0BjAbmzxRqLaCLInOiq/view?usp=sharing
2. Build a transfer learnt model from a model trained on EfficientNet/VGG by unfreezing some existing layers: https://drive.google.com/file/d/1DqBrEuvIdj4NbMinfqMCT2VNQx-l2Jc6/view?usp=sharing
3. Build a CNN model to identify the expression given the face: https://drive.google.com/file/d/1F3f4zZcztLuSDEvbc3zEqzK0Cb31oL1i/view?usp=sharing
4. A DCGAN model: https://drive.google.com/file/d/1O2AsEa1ESg1sMUR6YJOhcGN2OBNj5EQV/view?usp=sharing
5. VAE model: https://drive.google.com/file/d/1UB0yX8DGTr_eBIpeQI4LISDH9rRqjwSv/view?usp=sharing

# Process of Running
1. Download all files.
2. Install requirements "pip install -r requirements.txt"
3. Download 5 models in .h5 format from links above.
4. Edit app.py based on paths of models.
5. Run app.py using "streamlit run app.py".

# Process of Deployment
1. Add main_app.ipynb in Google Colab.
2. Add app.py file in drive.
3. Add models in h5 files in own drive.
4. Run first cell in main_app.ipynb and copy outputs.
!wget -q -O - ipv4.icanhazip.com
5. Run second cell in main_app.ipynb
!streamlit run app.py & npx localtunnel --port 8501
6. Click on link provided by your url in 5.
7. Type tunnel password from output of 4.
