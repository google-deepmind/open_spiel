#ifndef GAME_H_
#define GAME_H_

#include "jni.h"
#include <string>
#include "mode.h"
#include "moves.h"
#include "move.h"

class Context;

class Game{

public:

	Game(JNIEnv *env, jobject game, std::string game_path);

	std::string GetPath();

	jobject GetObj();

	void Create(int viewSize);

	std::string GetName();

	int StateFlags();

	Mode GetMode();

	void Start(Context context);

	Moves GetMoves(Context context);

	Move Apply(Context context, Move move);

private:	

	JNIEnv *env;
	jobject game;
	std::string game_path;

};

#endif