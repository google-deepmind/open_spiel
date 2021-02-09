add_library(stockfish OBJECT
        stockfish/src/benchmark.cpp
        stockfish/src/bitbase.cpp
        stockfish/src/bitboard.cpp
        stockfish/src/bitboard.h
        stockfish/src/endgame.cpp
        stockfish/src/endgame.h
        stockfish/src/evaluate.cpp
        stockfish/src/evaluate.h
        stockfish/src/material.cpp
        stockfish/src/material.h
        stockfish/src/misc.cpp
        stockfish/src/misc.h
        stockfish/src/movegen.cpp
        stockfish/src/movegen.h
        stockfish/src/movepick.cpp
        stockfish/src/movepick.h
        stockfish/src/pawns.cpp
        stockfish/src/pawns.h
        stockfish/src/position.cpp
        stockfish/src/position.h
        stockfish/src/psqt.cpp
        stockfish/src/psqt.h
        stockfish/src/search.cpp
        stockfish/src/search.h
        stockfish/src/thread.cpp
        stockfish/src/thread.h
        stockfish/src/thread_win32_osx.h
        stockfish/src/timeman.cpp
        stockfish/src/timeman.h
        stockfish/src/tt.cpp
        stockfish/src/tt.h
        stockfish/src/tune.cpp
        stockfish/src/tune.h
        stockfish/src/types.h
        stockfish/src/uci.cpp
        stockfish/src/uci.h
        stockfish/src/ucioption.cpp
        stockfish/src/incbin/incbin.h
        stockfish/src/syzygy/tbprobe.cpp
        stockfish/src/syzygy/tbprobe.h
        stockfish/src/nnue/evaluate_nnue.cpp
        stockfish/src/nnue/evaluate_nnue.h
        stockfish/src/nnue/nnue_accumulator.h
        stockfish/src/nnue/nnue_architecture.h
        stockfish/src/nnue/nnue_common.h
        stockfish/src/nnue/nnue_feature_transformer.h
        stockfish/src/nnue/architectures/halfkp_256x2-32-32.h
        stockfish/src/nnue/features/feature_set.h
        stockfish/src/nnue/features/features_common.h
        stockfish/src/nnue/features/half_kp.cpp
        stockfish/src/nnue/features/half_kp.h
        stockfish/src/nnue/features/index_list.h
        stockfish/src/nnue/layers/affine_transform.h
        stockfish/src/nnue/layers/clipped_relu.h
        stockfish/src/nnue/layers/input_slice.h
        stockfish_bot.cc
        stockfish_bot.h
        )

target_include_directories (stockfish PUBLIC stockfish/src)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DNNUE_EMBEDDING_OFF")

add_executable (stockfish_bot_test stockfish_bot_test.cc ${OPEN_SPIEL_OBJECTS}
        $<TARGET_OBJECTS:tests>)
add_test(stockfish_bot_test stockfish_bot_test)

add_executable (stockfish_uci stockfish/src/main.cpp ${OPEN_SPIEL_OBJECTS}
        $<TARGET_OBJECTS:tests>)