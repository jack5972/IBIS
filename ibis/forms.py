from django import forms

class BirdForm(forms.Form):
    size = forms.IntegerField(label="Size")
    colour = forms.CharField(label="Colour", max_length=100)
    bill = forms.CharField(label="Bill", max_length=100)
    head = forms.CharField(label="Head", max_length=100)
    upperwingCoverts = forms.CharField(label="Upperwing Coverts", max_length=100)
    underwingCoverts = forms.CharField(label="Underwisng Coverts", max_length=100)
    undertailCoverts = forms.CharField(label="Undertail Coverts", max_length=100)
    legs = forms.CharField(label="Legs", max_length=100)
    neck = forms.CharField(label="Neck", max_length=100)
    breast = forms.CharField(label="Breast", max_length=100)
