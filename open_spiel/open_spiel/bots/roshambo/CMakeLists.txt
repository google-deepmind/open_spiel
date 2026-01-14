add_library(roshambo OBJECT
  roshambo/BotClasses/actr_lag2_decay.h
  roshambo/BotClasses/adddriftbot2.h
  roshambo/BotClasses/addshiftbot3.h
  roshambo/BotClasses/antiflatbot.h
  roshambo/BotClasses/antirotnbot.h
  roshambo/BotClasses/biopic.h
  roshambo/BotClasses/boom.h
  roshambo/BotClasses/copybot.h
  roshambo/BotClasses/debruijn81.h
  roshambo/BotClasses/driftbot.h
  roshambo/BotClasses/flatbot3.h
  roshambo/BotClasses/foxtrotbot.h
  roshambo/BotClasses/freqbot.h
  roshambo/BotClasses/granite.h
  roshambo/BotClasses/greenberg.h
  roshambo/BotClasses/halbot.h
  roshambo/BotClasses/inocencio.h
  roshambo/BotClasses/iocainebot.h
  roshambo/BotClasses/marble.h
  roshambo/BotClasses/markov5.h
  roshambo/BotClasses/mixed_strategy.h
  roshambo/BotClasses/mod1bot.h
  roshambo/BotClasses/multibot.cc
  roshambo/BotClasses/multibot.h
  roshambo/BotClasses/peterbot.h
  roshambo/BotClasses/phasenbott.cc
  roshambo/BotClasses/phasenbott.h
  roshambo/BotClasses/pibot.h
  roshambo/BotClasses/piedra.h
  roshambo/BotClasses/predbot.h
  roshambo/BotClasses/r226bot.h
  roshambo/BotClasses/randbot.h
  roshambo/BotClasses/robertot.h
  roshambo/BotClasses/rockbot.h
  roshambo/BotClasses/rotatebot.h
  roshambo/BotClasses/rsb_bot.h
  roshambo/BotClasses/russrocker4.h
  roshambo/BotClasses/shofar.cc
  roshambo/BotClasses/shofar.h
  roshambo/BotClasses/suncrazybot.h
  roshambo/BotClasses/sunnervebot.h
  roshambo/BotClasses/sweetrock.h
  roshambo/BotClasses/switchalot.h
  roshambo/BotClasses/switchbot.h
  roshambo/BotClasses/textbot.h
  roshambo/BotClasses/zqmove.h
  roshambo/bot_map.cc
  roshambo/bot_map.h
  roshambo_bot.cc
  roshambo_bot.h
)
target_compile_options(roshambo PUBLIC -w)
target_include_directories(roshambo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(roshambo_bot_test roshambo_bot_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(roshambo_bot_test roshambo_bot_test)
