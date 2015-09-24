#ifndef GAME_OF_LIFE_H__
#define GAME_OF_LIFE_H__

extern "C" void runGameOfLifeIteration(unsigned char* d_src, unsigned char* d_dst, const size_t width, const size_t height);

#endif // GAME_OF_LIFE_H__
