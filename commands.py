import random

import click
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

from acme.auth import Auth
from app import app, db

# Models
from models.user import User
from models.recipe import Recipe

from faker import Factory, Faker
fake = Faker()

migrate = Migrate(app, db)
manager = Manager(app)
manager.add_command('db', MigrateCommand)


@app.cli.command()
def filldb():
    """Initialize the database."""
    click.echo('Fill the db')
    clear_db()
    add_users()
    add_recipes()
    db.session.commit()


def clear_db():
    Recipe.query.delete()
    User.query.delete()
    db.session.commit()


def add_users(n=5):
    #add admin
    admin = User(first_name='Morgun', last_name='Bezglazov', email='admin@gmail.com', password=Auth.crypt_password('admin'))
    db.session.add(admin)

    for i in range(n):
        db.session.add(User(
            first_name=fake.first_name(),
            last_name=fake.last_name(),
            email=fake.email(),
            password=Auth.crypt_password('admin')
        ))


def add_recipes(n=100):
    for i in range(n):
        db.session.add(Recipe(
            name=fake.sentence(),
            description=fake.text(),
            user_id=random.choice(User.query.all()).id
        ))

if __name__ == '__main__':
    manager.run()