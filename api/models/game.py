import abc
from enum import Enum


NUM_FINGERS = 5


class Hand(Enum):
    LEFT = 1
    RIGHT = 2


class Player:
    def __init__(self, name: str) -> None:
        self.name = name
        self.hands = {
            Hand.LEFT: 1,
            Hand.RIGHT: 1,
        }

    def is_defeated(self) -> bool:
        return self.sum_fingers() <= 0

    def sum_fingers(self) -> int:
        return self.hands[Hand.LEFT] + self.hands[Hand.RIGHT]


class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def satisfied_by(self, m: 'Movement', this: Player, opponent: Player) -> bool:
        pass


class AttackActiveHandsPolicy(Policy):
    def satisfied_by(self, m: 'AttackMovement', this: Player, opponent: Player) -> bool:
        if this.hands[m.this_hand] == 0:
            return False
        if opponent.hands[m.opponent_hand] == 0:
            return False
        return True


class RearrangeToConstantSumPolicy(Policy):
    def satisfied_by(self, m: 'RearrangeMovement', this: Player, opponent: Player) -> bool:
        if this.sum_fingers() != m.after[Hand.LEFT] + m.after[Hand.RIGHT]:
            return False
        return True


class RearrangeWithinNumFingersPolicy(Policy):
    def satisfied_by(self, m: 'RearrangeMovement', this: Player, opponent: Player) -> bool:
        if m.after[Hand.LEFT] < 0:
            return False
        if m.after[Hand.RIGHT] < 0:
            return False
        if m.after[Hand.LEFT] >= NUM_FINGERS:
            return False
        if m.after[Hand.RIGHT] >= NUM_FINGERS:
            return False


class RearrangeAsymmetryPolicy(Policy):
    def satisfied_by(self, m: 'RearrangeMovement', this: Player, opponent: Player) -> bool:
        if this.hands[Hand.LEFT] == m.after[Hand.LEFT] and this.hands[Hand.RIGHT] == m.after[Hand.RIGHT]:
            return False
        if this.hands[Hand.LEFT] == m.after[Hand.RIGHT] and this.hands[Hand.RIGHT] == m.after[Hand.LEFT]:
            return False
        return True


class Movement(metaclass=abc.ABCMeta):
    policies: list[Policy]

    def plan(self, this: Player, opponent: Player) -> bool:
        for policy in self.policies:
            if not policy.satisfied_by(self, this, opponent):
                return False
        return True

    @abc.abstractmethod
    def apply(self, this: Player, opponent: Player) -> None:
        pass


class AttackMovement(Movement):
    policies = [AttackActiveHandsPolicy()]

    def __init__(self, this_hand: Hand, opponent_hand: Hand) -> None:
        self.this_hand = this_hand
        self.opponent_hand = opponent_hand

    def apply(self, this: Player, opponent: Player) -> None:
        sm = opponent.hands[self.opponent_hand] + this.hands[self.this_hand]
        if sm < NUM_FINGERS:
            opponent.hands[self.opponent_hand] = sm
        else:
            opponent.hands[self.opponent_hand] = 0


class RearrangeMovement(Movement):
    policies = [RearrangeToConstantSumPolicy(), RearrangeAsymmetryPolicy()]

    def __init__(self, after: dict[Hand, int]) -> None:
        self.after = after

    def apply(self, this: Player, opponent: Player) -> None:
        this.hands = self.after


REARRANGE_MOVEMENTS: dict[int, 'RearrangeMovement'] = {
    (left + right * NUM_FINGERS): RearrangeMovement({
        Hand.LEFT: left,
        Hand.RIGHT: right,
    }) for left in range(NUM_FINGERS) for right in range(NUM_FINGERS)
}
ATTACK_MOVEMENTS: dict[int, 'AttackMovement'] = {
    len(REARRANGE_MOVEMENTS): AttackMovement(Hand.LEFT, Hand.LEFT),
    len(REARRANGE_MOVEMENTS) + 1: AttackMovement(Hand.LEFT, Hand.RIGHT),
    len(REARRANGE_MOVEMENTS) + 2: AttackMovement(Hand.RIGHT, Hand.LEFT),
    len(REARRANGE_MOVEMENTS) + 3: AttackMovement(Hand.RIGHT, Hand.RIGHT),
}
MOVEMENTS = REARRANGE_MOVEMENTS | ATTACK_MOVEMENTS


class Game:
    def __init__(self, me: Player, enemy: Player) -> None:
        self.__who_won: str | None = None
        self.__me = me
        self.__enemy = enemy
        self.__current_turn = 0

    def play_turn(self, m: Movement) -> bool:
        assert self.__who_won is None
        this, opponent = self.__this(), self.__opponent()
        assert m.plan(this, opponent)
        self.__increment_turn()
        m.apply(this, opponent)
        if opponent.is_defeated():
            self.__who_won = this.name
            return False
        return True

    def __increment_turn(self) -> None:
        self.__current_turn += 1

    def list_possible_movements(self) -> dict[int, Movement]:
        movements = {}
        for i, m in MOVEMENTS.items():
            if m.plan(self.__this(), self.__opponent()):
                movements[i] = m
        return movements

    def display(self) -> None:
        my_left_map = {
            0: '_____',
            1: '___|_',
            2: '__||_',
            3: '_|||_',
            4: '||||_',
            5: '||||/',
        }
        my_right_map = {k: v[::-1] for k, v in my_left_map.items()}
        enemy_right_map = {k: v.replace('_', '-').replace('/', '\\') for k, v in my_left_map.items()}
        enemy_left_map = {k: v[::-1] for k, v in enemy_right_map.items()}

        print(f'Current Turn: {self.__current_turn:03d}')
        print()
        print('  RIGHT  LEFT   ')
        print(f'  {enemy_right_map[self.__enemy.hands[Hand.RIGHT]]}  {enemy_left_map[self.__enemy.hands[Hand.LEFT]]}  ')
        print()
        print(f'  {my_left_map[self.__me.hands[Hand.LEFT]]}  {my_right_map[self.__me.hands[Hand.RIGHT]]}  ')
        print('  LEFT   RIGHT  ')
        print('----------------')

    def current_my_turn(self) -> bool:
        return self.__current_turn % 2 == 0

    def current_enemy_turn(self) -> bool:
        return self.__current_turn % 2 == 1

    def __this(self) -> Player:
        return self.__me if self.current_my_turn() else self.__enemy

    def __opponent(self) -> Player:
        return self.__enemy if self.current_my_turn() else self.__me

    @property
    def me(self) -> Player:
        return self.__me

    @property
    def enemy(self) -> Player:
        return self.__enemy

    @property
    def who_won(self) -> str:
        assert self.__who_won is not None
        return self.__who_won


if __name__ == "__main__":
    import random

    game = Game(Player('me'), Player('enemy'))
    game.display()

    cont = True
    while cont:
        choices = list(game.list_possible_movements().values())
        choice = random.choice(choices)
        cont = game.play_turn(choice)
        game.display()

    print(f'"{game.who_won}" won')
