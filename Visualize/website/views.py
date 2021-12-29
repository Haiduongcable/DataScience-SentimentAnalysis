from flask import Blueprint, render_template, request, flash, jsonify
# from flask_login import login_required, current_user
# from .models import Note
# from . import db
import json
from .Sentiment_Analysis import Sentiment_Analysis
from .utils import show_pie_char
views = Blueprint('views', __name__)
sentiment_analysis = Sentiment_Analysis(2)


# link sanpham loi 
# khong co review


@views.route('/', methods=['GET', 'POST'])
def home():
    notes = []
    count_negative = ''
    count_review = ''
    if request.method == 'POST':
        review = request.form.get('note')
        if len(review) < 1:
            flash('Review is too short!', category='error')
        
        l_class = ["Negative", "Positive"]
        type_input, result = sentiment_analysis.inference(review)
        if type_input == "text":
            class_predict, success = result
            if not success:
                flash('Review error syntax!', category='error')
            elif class_predict == 0:
                flash('Negative!', category='error')
            elif class_predict == 1:
                flash('Positive!', category='success')
            
        else:
            l_predict, l_used_review = result
            count_negative = 0
            for index, predict_value in enumerate(l_predict):
                if predict_value == 0:
                    count_negative += 1
                    notes.append(l_used_review[index])
            count_review = len(l_predict)
            show_pie_char(count_negative,count_review)
            count_negative = str(count_negative)
            count_review = str(count_review)
            
    return render_template("home.html", notes=notes, num_negative = count_negative, num_review = count_review)


# @views.route('/delete-note', methods=['POST'])
# def delete_note():
#     note = json.loads(request.data)
#     noteId = note['noteId']
#     note = Note.query.get(noteId)
#     if note:
#         if note.user_id == current_user.id:
#             db.session.delete(note)
#             db.session.commit()

#     return jsonify({})
