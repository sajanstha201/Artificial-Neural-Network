from django.shortcuts import render,redirect
from src.pipeline.predict_pipeline import prediction
def home(request):
    pred=0
    if(request.method=='POST'):
        data=request.POST
        pred=prediction(data)
        print("sajan shrestha",pred)
        pred=100
        return redirect('/titanic/')
    return render(request,'index.html',{'living_percentage':pred})