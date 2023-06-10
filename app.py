from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from inference import predict_wave
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from flask_mail import Message, Mail
from flask import flash
import secrets
from flask import Flask, request, render_template, redirect, url_for
from datetime import datetime, timedelta
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'C:/Users/ELZAHBIA/Desktop/FLASK CAR PROBLEM/Sounds'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['SECURITY_PASSWORD_SALT'] = 'your_salt_here'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True


db = SQLAlchemy(app)
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(80), nullable=False)
    lastname = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    carname = db.Column(db.String(80), nullable=False)
    caryear = db.Column(db.Integer, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<User {self.email}>'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

    def __repr__(self):
        return f'<Role {self.name}>'


# Create the database tables
with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return render_template('sign_up.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    result = None
    if request.method == 'POST':
        # Get the file from the request object
        file = request.files['audio']
        # Save the file to the uploads folder
        file_sound = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_sound)
        # Detect the car problem from the uploaded audio file
        result = predict_wave(file_sound)
        print(result)
    # Render the upload.html template with the result as a context variable
    return render_template('upload.html', result=result)


@app.route('/account', methods=['GET', 'POST'])
def account():
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        if request.method == 'POST':
            user.firstname = request.form['firstname']
            user.lastname = request.form['lastname']
            user.email = request.form['email']
            user.phone = request.form['phone']
            user.carname = request.form['carname']
            user.caryear = request.form['caryear']
            db.session.commit()
            flash('Your account information has been updated.')
            return redirect(url_for('account'))
        return render_template('account.html', user=user)
    else:
        return redirect(url_for('login'))
    
@app.route('/sign_up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        carname = request.form['carname']
        caryear = request.form['caryear']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user is not None:
            return 'Email already exists!'
        new_user = User(firstname=firstname, lastname=lastname, email=email, phone=phone, carname=carname,
                        caryear=caryear)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('sign_up.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user is None:
            return 'Email does not exist!'
        if not user.check_password(password):
            return 'Password incorrect!'
        session['user_id'] = user.id
        return redirect(url_for('upload'))
    return render_template('login.html')

from flask import Flask, request, render_template, redirect, url_for
from datetime import datetime, timedelta
import random
import os
from twilio.rest import Client

# Set environment variables for your credentials
# Read more at http://twil.io/secure
account_sid = "ACff3ed3786f254be01ac00b807ce297e9"
auth_token = "6d11be79727f30ad38371792d8fd55fc"
verify_sid = "VA806b85460e2d73b164625b95e38c73d2"
verified_number = "+201010634589"

client = Client(account_sid, auth_token)
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        phone = request.form['phone']
        user = User.query.filter_by(phone=phone).first()
        if user:
            # Generate a verification code
            verification_code = str(random.randint(100000, 999999))
            # Store the verification code in the database or storage
            set_verification_code(phone, verification_code)
            # Send a verification code via SMS
            verification = client.verify.services(verify_sid).verifications.create(
                to=phone,
                channel='sms'
            )
            # Store the verification SID, phone number, and token in the session
            session['verification_sid'] = verification.sid
            session['phone'] = phone
            session['token'] = serializer.dumps(user.email, salt='password-reset')
            # Redirect to the verification page
            return redirect(url_for('verify_code', phone=session['phone']))
        flash('If your phone number is registered with us, you will receive an SMS with instructions to reset your password.')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')
# Define a dictionary to store verification codes
verification_codes = {}

# Define the get_verification_code function to retrieve the verification code from the dictionary
def get_verification_code(phone):
    return verification_codes.get(phone)

# Define the set_verification_code function to set the verification code in the dictionary
def set_verification_code(phone, code):
    verification_codes[phone] = code


def is_verification_code_valid(code, input_code):
    # Check if the input code matches the generated code
    if code == input_code:
        return True
    else:
        return False
def delete_verification_code(code):
    # Delete the verification code from the database or storage
    # Here, we're just printing a message to indicate that the code has been deleted
    print(f"Verification code {code} has been deleted")
def generate_token():
    """Generate a unique token for password reset requests."""
    token = secrets.token_urlsafe(32)
    return token

@app.route('/verify_code/<phone>', methods=['GET', 'POST'])
def verify_code(phone):
    # Get the verification code from the database
    verification_code = get_verification_code(phone)

    if request.method == 'POST':
        # If the request method is POST, get the code entered by the user
        code = request.form['code']

        if is_verification_code_valid(verification_code, code):
            # If the code entered by the user matches the verification code, delete the verification code from the database
            delete_verification_code(verification_code)

            # Set the token variable in the session and redirect to the reset_password route
            token = generate_token()
            session['token'] = token
            return redirect(url_for('reset_password', token=token))

        else:
            # If the code entered by the user is invalid, return an error message
            flash('Invalid verification code.')
            return redirect(url_for('verify_code', phone=phone))

    # If the request method is GET, render the verify_code.html template with the phone number passed as a parameter in the URL
    return render_template('verify_code.html', phone=phone)


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if 'token' not in session or session['token'] != token:
        flash('Invalid or expired token.')
        return redirect(url_for('forgot_password'))

    user = User.query.filter_by(phone=session['phone_number']).first()

    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('reset_password', token=token))

        user.password = generate_password_hash(password)
        db.session.commit()

        flash('Your password has been reset.')
        return redirect(url_for('login'))

    return render_template('reset_password.html')



@app.route('/logout')
def logout():
    # Clear the user's session
    session.clear()
    # Redirect the user to the login page
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)