from flask import Blueprint, render_template, request, flash, jsonify
# from flask_login import login_required, current_user
# from .models import Note
# from . import db
import json
from .Sentiment_Analysis import Sentiment_Analysis
views = Blueprint('views', __name__)
sentiment_analysis = Sentiment_Analysis(2)

@views.route('/', methods=['GET', 'POST'])
def home():
    current_user = ''
    if request.method == 'POST':
        review = request.form.get('note')
        l_class = ["Negative", "Positive"]
        class_predict, success = sentiment_analysis.inference(review)
        
        if len(review) < 1:
            flash('Review is too short!', category='error')
        elif not success:
            flash('Review error syntax!', category='error')
        elif class_predict == 0:
            flash('Negative!', category='error')
        elif class_predict == 1:
            flash('Positive!', category='success')

    return render_template("home.html", user=current_user)


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
