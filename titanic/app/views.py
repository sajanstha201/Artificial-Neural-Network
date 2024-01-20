from django.shortcuts import render,redirect
from src.pipeline.predict_pipeline import prediction
def home(request):
    pred=None
    if(request.method=='POST'):
        data=request.POST
        pred=prediction(data)
    return render(request,'index.html',{'living_percentage':pred})