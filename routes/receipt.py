from flask import render_template
from flask import request

from app import app
from models.recipe import Recipe

@app.route('/receipt/<int:id>',methods=['GET'])
def receipt(id):
    receipt = Recipe.query.filter_by(id=id).first()
    return render_template('receipts/show.html', receipt=receipt)