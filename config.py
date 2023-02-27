from datetime import date

MODEL_NAME = "Winner_based_on_first_set"
SAVE_CSV_TO = f"./{MODEL_NAME}_{date.today()}/"
MAX_SET_NO = 2
LIST_OF_FEATURES = [
    "winner",
    "slam",
    "MAX(points.SetWinner)",
    "MEAN(points.Rally)",
    "STD(points.Rally)",
    "SUM(points.P1Ace)",
    "SUM(points.P1BreakPoint)",
    "SUM(points.P1BreakPointMissed)",
    "SUM(points.P1BreakPointWon)",
    "SUM(points.P1DoubleFault)",
    "MEAN(points.P1FirstSrvIn)",
    "SUM(points.P1ForcedError)",
    "MAX(points.P1GamesWon)",
    "MAX(points.P1Momentum)",
    "SUM(points.P1NetPoint)",
    "SUM(points.P1NetPointWon)",
    "MAX(points.P1PointsWon)",
    "SUM(points.P1UnfErr)",
    "SUM(points.P1Winner)",
    "SUM(points.P2Ace)",
    "SUM(points.P2BreakPoint)",
    "SUM(points.P2BreakPointMissed)",
    "SUM(points.P2BreakPointWon)",
    "SUM(points.P2DoubleFault)",
    "MEAN(points.P2FirstSrvIn)",
    "SUM(points.P2ForcedError)",
    "MAX(points.P2GamesWon)",
    "MAX(points.P2Momentum)",
    "SUM(points.P2NetPoint)",
    "SUM(points.P2NetPointWon)",
    "MAX(points.P2PointsWon)",
    "SUM(points.P2UnfErr)",
    "SUM(points.P2Winner)"
]