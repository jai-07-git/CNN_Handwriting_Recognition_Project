"""
Google sign-in. Uses Authlib for the OAuth2/OpenID Connect handshake and
Flask-Login to track the signed-in user across requests via a signed session
cookie. No passwords are ever stored — identity comes entirely from Google.
"""
from authlib.integrations.flask_client import OAuth
from flask import Blueprint, redirect, render_template, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user

from database import get_user_by_id, upsert_user

login_manager = LoginManager()
login_manager.login_view = "auth.login_page"
oauth = OAuth()
auth_bp = Blueprint("auth", __name__)


class User(UserMixin):
    def __init__(self, id, email, name, picture):
        self.id = id
        self.email = email
        self.name = name
        self.picture = picture

    @staticmethod
    def from_row(row):
        if row is None:
            return None
        return User(id=row["id"], email=row["email"], name=row["name"], picture=row["picture"])


def init_auth(app):
    login_manager.init_app(app)
    oauth.init_app(app)
    oauth.register(
        name="google",
        client_id=app.config["GOOGLE_CLIENT_ID"],
        client_secret=app.config["GOOGLE_CLIENT_SECRET"],
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    app.register_blueprint(auth_bp)


@login_manager.user_loader
def load_user(user_id):
    return User.from_row(get_user_by_id(user_id))


@auth_bp.route("/login")
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    return render_template("login.html")


@auth_bp.route("/login/google")
def login_google():
    redirect_uri = url_for("auth.google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/login/google/callback")
def google_callback():
    token = oauth.google.authorize_access_token()
    userinfo = token.get("userinfo")
    if not userinfo:
        userinfo = oauth.google.userinfo(token=token)

    google_sub = userinfo["sub"]
    email = userinfo.get("email", "")
    name = userinfo.get("name") or email or "Unknown"
    picture = userinfo.get("picture", "")

    row = upsert_user(user_id=google_sub, email=email, name=name, picture=picture)
    login_user(User.from_row(row))
    return redirect(url_for("index"))


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login_page"))