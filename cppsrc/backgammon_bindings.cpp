/*backgammon_bindings.cpp*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "game.hpp"
#include "player.hpp"
#include "pieces.hpp"

namespace py = pybind11;

std::pair<bool, int> gameOverStatus(Game &game)
{
    int winner = -1;
    bool isOver = game.over(&winner);
    return std::make_pair(isOver, winner);
}

std::pair<bool, std::string> tryMoveWrapper(Game &game, Player *player, int dice, int origin, int dest)
{
    std::string err;
    bool success = game.tryMove(player, dice, origin, dest, err);
    return std::make_pair(success, err);
}

PYBIND11_MODULE(backgammon_env, m)
{
    m.doc() = "Backgammon game environment for Reinforcement Learning";

    // Player enum
    py::enum_<Player::PLAYERS>(m, "PlayerType")
        .value("PLAYER1", Player::PLAYER1)
        .value("PLAYER2", Player::PLAYER2);

    // Player class
    py::class_<Player>(m, "Player")
        .def(py::init<const std::string &, Player::PLAYERS>())
        .def("getName", &Player::getName)
        .def("getNum", &Player::getNum);

    // Pieces class
    py::class_<Pieces>(m, "Pieces")
        .def("numJailed", &Pieces::numJailed)
        .def("numFreed", &Pieces::numFreed);

    // Game class - main interface for RL
    py::class_<Game>(m, "Game")
        .def(py::init<int>())
        .def("setPlayers", &Game::setPlayers)
        .def("getTurn", &Game::getTurn)
        .def("setTurn", &Game::setTurn)
        .def("getGameBoard", &Game::getGameBoard)
        .def("getPieces", &Game::getPieces, py::return_value_policy::reference)

        // Core RL methods
        .def("legalMoves", &Game::legalMoves)
        .def("legalTurnSequences", &Game::legalTurnSequences)
        .def("tryMove", &tryMoveWrapper)
        .def("is_game_over", &gameOverStatus)
        .def("clone", &Game::clone)

        // Utility methods
        .def("getJailedCount", &Game::getJailedCount)
        .def("getBornOffCount", &Game::getBornOffCount)
        .def("printGameBoard", &Game::printGameBoard)
        .def("reset", &Game::populateBoard)
        .def("populateBoard", &Game::populateBoard)
        .def("roll_dice", &Game::rollDice,
             "Roll two dice and return an array [die1, die2]")
        .def("get_last_dice", &Game::getLastDice,
             "Return the most recently rolled dice as [die1, die2]");
}