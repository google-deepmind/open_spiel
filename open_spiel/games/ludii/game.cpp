#include "game.h"
#include "context.h"

Game::Game(JNIEnv *env, jobject game, std::string game_path)
    : env(env)
    , game(game)
    , game_path(game_path)
{}

std::string Game::GetPath() const
{
    return game_path;
}

jobject Game::GetObj() const
{
    return game;
}

std::string Game::GetName() const
{

    jclass gameClass = env->FindClass("game/Game");
    jmethodID name_id = env->GetMethodID(gameClass, "name", "()Ljava/lang/String;");
    jstring stringArray = (jstring) env->CallObjectMethod(game, name_id);

    //convert jstring game name to char array
    const char *strReturn = env->GetStringUTFChars(stringArray, 0);
    std::string string_name(strReturn);
    env->ReleaseStringUTFChars(stringArray, strReturn);

    return string_name;
}

void Game::Create(int viewSize) const
{
    jclass gameClass = env->FindClass("game/Game");
    jmethodID create_id = env->GetMethodID(gameClass, "create", "(I)V");
    env->CallVoidMethod(game, create_id, viewSize);
}

int Game::StateFlags() const
{
    jclass gameClass = env->FindClass("game/Game");
    jmethodID stateFlags_id = env->GetMethodID(gameClass, "stateFlags", "()I");
    return (int) env->CallIntMethod(game, stateFlags_id);
}

Mode Game::GetMode() const
{
    jclass gameClass = env->FindClass("game/Game");
    jmethodID mode_id = env->GetMethodID(gameClass, "mode", "()Lgame/mode/Mode;");
    jobject mode = env->CallObjectMethod(game, mode_id);
    return Mode(env, mode);
}

void Game::Start(Context context) const
{

    jclass gameClass = env->FindClass("game/Game");
    jmethodID start_id = env->GetMethodID(gameClass, "start", "(Lutil/Context;)V");
    env->CallVoidMethod(game, start_id, context.GetObj());
}

Moves Game::GetMoves(Context context) const
{

    jclass gameClass = env->FindClass("game/Game");
    jmethodID moves_id = env->GetMethodID(gameClass, "moves", "(Lutil/Context;)Lgame/rules/play/moves/Moves;");
    jobject moves_obj = env->CallObjectMethod(game, moves_id, context.GetObj());
    jclass clsObj = env->GetObjectClass(context.GetObj());

    return Moves(env, moves_obj);
}

Move Game::Apply(Context context, Move move) const
{

    jclass gameClass = env->FindClass("game/Game");
    jmethodID apply_id = env->GetMethodID(gameClass, "apply", "(Lutil/Context;Lutil/Move;)Lutil/Move;");
    jobject move_obj = env->CallObjectMethod(game, apply_id, context.GetObj(), move.GetObj());

    return Move(env, move_obj);
}
