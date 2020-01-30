import random
import torch

import config
import constants

from showdown.engine.find_state_instructions import update_attacking_move, lookup_move
from showdown.battle_bots.helpers import format_decision
from showdown.websocket_client import PSWebsocketClient
from showdown.run_battle import start_bot_battle

type_dict = {
    'normal': 0,
    'fighting': 1,
    'flying': 2,
    'poison': 3,
    'ground': 4,
    'rock': 5,
    'bug': 6,
    'ghost': 7,
    'steel': 8,
    'fire': 9,
    'water': 10,
    'grass': 11,
    'electric': 12,
    'psychic': 13,
    'ice': 14,
    'dragon': 15,
    'dark': 16,
    'fairy': 17
}

status_dict = {
    'brn': 0,
    'slp': 1,
    'par': 2,
    'psn': 3,
    'tox': 4,
    'frz': 5
}


class PokemonEnvManager():
    def __init__(self, device):
        self.device = device
        self.status_factor = 0.4
        self.fainted_factor = 1
        self.is_first_move = False
        self.done = True
        self.numMoves = 0
        self.winner = 0

    async def initbots(self):
        self.bots = [
            {
                'battle': None,
                'ps_websocket_client': await PSWebsocketClient.create(config.bot_username_1, config.bot_password_1, config.websocket_uri),
                'switch': False
            },
            {
                'battle': None,
                'ps_websocket_client': await PSWebsocketClient.create(config.bot_username_2, config.bot_password_2, config.websocket_uri),
                'switch': False
            }
        ]
        await self.bots[0]['ps_websocket_client'].login()
        await self.bots[1]['ps_websocket_client'].login()

    async def start(self):
        await self.bots[0]['ps_websocket_client'].challenge_user(config.bot_username_2, 'gen8randombattle', {})
        await self.bots[1]['ps_websocket_client'].accept_challenge('gen8randombattle', {})
        self.bots[0]['battle'] = await start_bot_battle(self.bots[0]['ps_websocket_client'])
        self.bots[1]['battle'] = await start_bot_battle(self.bots[1]['ps_websocket_client'])
        self.is_first_move = True
        self.done = False
        self.numMoves = 0

    async def stop(self):
        await self.bots[0]['ps_websocket_client'].send_message(self.bots[0]['battle'].battle_tag, '/forfeit')
        await self.bots[0]['ps_websocket_client'].leave_battle(self.bots[0]['battle'].battle_tag, save_replay = False)
        await self.bots[1]['ps_websocket_client'].leave_battle(self.bots[1]['battle'].battle_tag, save_replay = False)
        self.bots[0]['battle'] = None
        self.bots[1]['battle'] = None
        self.done = True

    def get_type_array(self, type_list):
        type_array = [0 for i in range(18)]
        for type in type_list:
            if type != 'typeless':
                type_array[type_dict[type]] = 1

        return type_array

    # Accept move and return 24 length array
    def get_move_state(self, move, state, first_move):
        move_dict = update_attacking_move(
            state.self.active,
            state.opponent.active,
            lookup_move(move['id']),
            {},
            first_move,
            state.weather
        )
        # First 18 Fields: Type
        move_state = self.get_type_array([move_dict['type']])

        # One field for basePower
        move_state += [move_dict['basePower'] / 100]

        # One Field for accuracy
        if move_dict['accuracy'] == True:
            move_state += [4]
        else:
            move_state += [move_dict['accuracy'] / 100]

        # 3 Fields for category
        move_state += [
            int(move_dict['category'] == "physical"),
            int(move_dict['category'] == "special"),
            int(move_dict['category'] == "status")
        ]

        # One field for enabled
        move_state += [
            int(not move['disabled'])
        ]

        return move_state

    # Accept pokemon and return 31 long array
    def get_pokemon_state(self, pokemon):
        # 18 Fields for types
        pokemon_state = self.get_type_array(pokemon.types)

        # 6 Fields for stats
        pokemon_state += [
            pokemon.maxhp / 400,
            pokemon.attack / 400,
            pokemon.defense / 400,
            pokemon.special_attack / 400,
            pokemon.special_defense / 400,
            pokemon.speed / 400
        ]

        # 1 Field for current health
        pokemon_state += [
            pokemon.hp / pokemon.maxhp
        ]

        # 6 Fields for status
        pokemon_status = [0 for i in range(6)]
        if pokemon.status != None:
            pokemon_status[status_dict[pokemon.status]] = 1

        pokemon_state += pokemon_status

        return pokemon_state

    # 313 Long Array: State
    def get_bot_state(self, bot):
        battle_state = self.bots[bot]['battle'].create_state()

        # Active pokemon - 31
        bot_state = self.get_pokemon_state(battle_state.self.active)

        # Opponent pokemon - 31
        bot_state += self.get_pokemon_state(battle_state.opponent.active)

        # Moves - 4 * 24 = 96
        is_first_move = self.is_first_move or self.bots[bot]['switch']
        for move in battle_state.self.active.moves:
            bot_state += self.get_move_state(move, battle_state, is_first_move)

        bot_state += [0 for i in range((4 - len(battle_state.self.active.moves)) * 24)]


        # Team - 5 * 31 = 155
        for pokemon in battle_state.self.reserve.values():
            bot_state += self.get_pokemon_state(pokemon)

        bot_state += [0 for i in range((5 - len(battle_state.self.reserve.values())) * 24)]

        return torch.cuda.FloatTensor(bot_state).unsqueeze(0)

    def get_state(self, bot):
        return self.bots[bot]['battle'].create_state()

    def calculate_reward(self, prev_state, bot):
        current_state = self.bots[bot]['battle'].create_state()
        reward = 0
        my_health = 0
        my_old_health = 0
        my_fainted = 0
        my_old_fainted = 0
        my_status = 0
        my_old_status = 0
        opp_health = 0
        opp_old_health = 0
        opp_fainted = 0
        opp_old_fainted = 0
        opp_status = 0
        opp_old_status = 0

        my_old_pokemon = prev_state.self.reserve
        my_old_pokemon[prev_state.self.active.id] = prev_state.self.active

        my_pokemon = current_state.self.reserve
        my_pokemon[current_state.self.active.id] = current_state.self.active

        opp_old_pokemon = prev_state.opponent.reserve
        opp_old_pokemon[prev_state.opponent.active.id] = prev_state.opponent.active

        opp_pokemon = current_state.opponent.reserve
        opp_pokemon[current_state.opponent.active.id] = current_state.opponent.active

        for pokemon in my_pokemon.values():
            my_health += pokemon.hp / pokemon.maxhp
            my_fainted += pokemon.hp == 0
            my_status += pokemon.status != None

        for pokemon in my_old_pokemon.values():
            my_old_health += pokemon.hp / pokemon.maxhp
            my_old_fainted += pokemon.hp == 0
            my_old_status += pokemon.status != None

        for pokemon in opp_pokemon.values():
            opp_health += pokemon.hp / pokemon.maxhp
            opp_fainted += pokemon.hp == 0
            opp_status += pokemon.status != None

        for pokemon in opp_old_pokemon.values():
            opp_old_health += pokemon.hp / pokemon.maxhp
            opp_old_fainted += pokemon.hp == 0
            opp_old_status += pokemon.status != None

        reward -= (my_health - my_old_health) \
                + (my_status - my_old_status) * self.status_factor \
                + (my_fainted - my_old_fainted) * self.fainted_factor

        reward += (opp_health - opp_old_health) \
                + (opp_status - opp_old_status) * self.status_factor \
                + (opp_fainted - opp_old_fainted) * self.fainted_factor

        if my_fainted == 6:
            self.done = True
            reward -= 10

        if opp_fainted == 6:
            self.done = True
            reward += 10

        return torch.cuda.FloatTensor([reward])

    async def make_random_move(self, bot):
        my_options = self.bots[bot]['battle'].get_all_options()[0]

        my_move = format_decision(self.bots[bot]['battle'], my_options[random.randrange(len(my_options))])

        await self.bots[bot]['ps_websocket_client'].send_message(
            self.bots[bot]['battle'].battle_tag, my_move
        )

    async def make_move(self, state, result, bot):
        my_options = self.bots[bot]['battle'].get_all_options()[0]

        moves = []
        switches = []
        for option in my_options:
            if option.startswith(constants.SWITCH_STRING + " "):
                switches.append(option.split(' ')[1])
            else:
                moves.append(option)

        my_move = None
        action_id = None

        for top in result:
            if top <= 3:
                if len(state.self.active.moves) > top:
                    move = state.self.active.moves[top]['id']
                    if move in moves:
                        my_move = move
                        action_id = top
                        break
            else:
                switch_pokemon = list(state.self.reserve)[top - 4]
                if switch_pokemon in switches:
                    my_move = constants.SWITCH_STRING + " " + switch_pokemon
                    action_id = top
                    break
        else:
            my_move = moves[0]
            action_id = 0

        my_move = format_decision(self.bots[bot]['battle'], my_move)

        await self.bots[bot]['ps_websocket_client'].send_message(
            self.bots[bot]['battle'].battle_tag, my_move
        )

        self.is_first_movmy_movee = False

        return my_move, torch.cuda.LongTensor([action_id])
