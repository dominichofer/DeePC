# python-reversi

## Description
This is a Python library for Reversi.
It requires at least Python 3.10, and the libraries listed in pyproject.toml.

## Installation
Run `pip install .` to install the package and all its dependencies.

## Usage

It implements a class to represent a reversi position
```python
class Position:
	__init__(uint64, uint64)
	start() -> Position		
	from_string(str) -> Position
	
	player() -> uint64
	opponent() -> uint64
	discs() -> uint64
	empties() -> uint64
	empty_count() -> int
```

where uint64 values represent bitboards.
It names Fields through an enum
```python
enum Field:
	H8 ... A1, PS
```

and hosts several functions to operate on positions
```python
single_line_string(Position) -> str
multi_line_string(Position) -> str
possible_moves(Position) -> Iterable[Field]
play(Position, move: Field) -> Position
play_pass(Position) -> Position
is_game_over(Position) -> bool
end_score(Position) -> int
children(Position, plies: int) -> Iterable[Position]
unique_positions(Positions) -> Positions
```

A full or partial game from an arbitrary starting position is represented with
```python
class Game:
	from_string(str) -> Game
	start_position -> Position
	moves -> list[Field]
	positions() -> Iterable[Position]
	last_position() -> Position
	play(move: Field | str)
```

A scored position and game is represented by
```python
class ScoredPosition:
	pos: Position
	score: int
	from_string(str) -> ScoredPosition

class ScoredGame:
	game: Game
	scores: list[int]
	from_string(str) -> ScoredGame
```

To play a game or several games between two players, there is an interface
```python
class Player:
	choose_move(Position) -> Field
	choose_moves(Positions) -> list[Fields]
```

which is used for the functions

```python
played_game(first: Player, second: Player, start: Position) -> Game
played_game(first: Player, second: Player, starts: Positions) -> list[Game]
```

There's also a 'RandomPlayer' which chooses a random legal move.
To collect all positions from the various classes, there's 
```python
positions(arg) -> Iterable[Position]
```

which can act on Game, ScoredGame, ScoredPosition, Iterable[Games], Iterable[ScoredGame], Iterable[ScoredPosition].
The function
```python
scores(arg) -> Iterable[int]
```

extracts scores from ScoredGame, ScoredPosition, Iterable[ScoredGame], Iterable[ScoredPosition].
To extract or generate scored positions, there's
```python
scored_positions(ScoredGame) -> Iterable[ScoredPosition]
scored_positions(Iterable[ScoredGame]) -> Iterable[ScoredPosition]
scored_positions(Iterable[Position], Iterable[int]) -> Iterable[ScoredPosition]
```

Positions and ScoredPositions can then be filtered with
```python
empty_count_filtered(arg: Iterable, empty_count: int) -> Iterable
empty_count_range_filtered(arg: Iterable, lower: int, upper: int) -> Iterable
```

To conduct a search, several algorithms, wrapped in classes are available
```python
class NegaMax:
	nodes: int
	eval(Position) -> int

class AlphaBeta:
	nodes: int
	__init__(move_sorter)
	eval(Position, window: OpenInterval) -> int

class PrincipalVariation:
	nodes: int
	__init__(move_sorter, transposition_table, cutters)
	eval(Position, window: OpenInterval, Intensity) -> Result
```

where
```python
class Result:
	window: ClosedInterval
	intensity: Intensity
	best_move: Field
```

denotes the window of the true score as either
a fail low window [min_score, upper_limit],
a fail high window [lower_limit, max_score],
or an exact window [score, score];
the retreived intensity, as a search may return more than requested,
and the best move found.
To enhance a search, there's a hash table which can be used as a transposition table
```python
class HashTable:
	__init__(size: int)
	update(Position, Result)
	look_up(Position) -> Result
	clear()
```

and two move sorters
```python
sorted_by_mobility(Position, Moves)
sorted_by_mobility_and_tt(HashTable)
```

In addition to that, there are implementations for open and closed intervals
```python
class OpenInterval:
	int lower
	int upper
	
class ClosedInterval:
	int lower
	int upper
```

the maximal and minimal score in reversi
```python
max_score = +64
min_score = -64
```

and the intensity of a search
```python
class Intensity:
	depth: int
	confidence_level: float
```

To read and write files with positions, scored positions, games and scored games there's
```python
read_file(file_path) -> list
write_file(file_path, items: Iterable)
```

Additionally a position can be plotted as a png with
```python
png(Position) -> PIL.Image
```

which is shown by example in example.ipynb.

A position can be set up interactively from a GUI board, sting or hex values with
```python
setup_position()
```

In the folder `data` are a set of scored positions
```
endgame.pos
fforum-1-19.pos
fforum-20-39.pos
fforum-40-59.pis
fforum-60-79.pos
```

as well as scored games that arive through self play of Edax at diffrent levels, starting from all unique positions with 54 empty fields.
```
Edax4.4_level_1_vs_Edax4.4_level_1_from_e54
Edax4.4_level_5_vs_Edax4.4_level_5_from_e54
Edax4.4_level_10_vs_Edax4.4_level_10_from_e54
Edax4.4_level_15_vs_Edax4.4_level_15_from_e54
Edax4.4_level_20_vs_Edax4.4_level_20_from_e54
```

where all positions with less than 31 empty fields have been solved with Edax level=60, which takes hundreds of hours. The code to recreate them is in data_generation.ipynb.
