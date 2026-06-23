import synreal_sim as sim
import os
import json
from pathlib import Path

password_dir = Path(__file__).parent.resolve()


def log_in_simulation(login_file = password_dir / 'simulation_login.json'):
    """Log in to the SynReal simulation service.

    If ``login_file`` exists, credentials are loaded from the JSON file. The
    file should follow ``simulation_login_template.json`` and contain ``name``
    and ``pass_word`` fields. If the file is missing or ``login_file`` is
    ``None``, credentials are requested from the input prompt.

    Args:
        login_file: Path to a JSON credentials file. Defaults to
            ``style3d/simulation_login.json``.
    """

    name = ''

    if not sim.is_login():
        if login_file and os.path.exists(login_file):
            with open(login_file,'r') as f:
                login=json.load(f)
                name = login['name']
                pass_word = login['pass_word']
        else:
            name = input('Enter your name : ')
            pass_word = input('Enter your password : ')

        sim.login(name, pass_word, True, None)

    if sim.is_login():
        print(f'login successful {name}')
    else:
        print('login failed')
