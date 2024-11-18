import streamlit as st
import pickle

model=pickle.load(open('spam.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))

st.title("Email Spam Classification Application")
st.write("this is a machine learning application")
st.subheader("Cassification")
user_input=st.text_area("Enter your email")


if st.button("Classify"):
	if user_input:
		data=[user_input]
		print(data)
		a=cv.transform(data).toarray()
		result=model.predict(a)
		if result[0]==0:
			st.success("This is Not A Spam Email")
		else:
			st.error("This is A Spam Email")
	else:
            st.write("Please enter an email to classify.")