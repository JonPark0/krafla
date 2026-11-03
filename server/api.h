/* OpenAI API requests: prompts from chat messages or completion prompts. */
#ifndef ML_API_H
#define ML_API_H
#include <stddef.h>
#include "model/tokenizer.h"
#include "server/json.h"

/* Token ids of a chat request's `messages` (node msgs) rendered with chat
 * template tpl, as ml_run --chat would feed them: system and developer
 * messages form the system prompt (default_system if there are none), each
 * user turn is rendered by ml_chat_render, an assistant reply follows its
 * turn and ends with the template's stop token. A final assistant message is
 * left open, for the model to continue. Returns 0 with *toks malloc'd, or -1
 * with a message in err. */
int srv_chat_tokens(ml_tokenizer *tok, int tpl, const char *default_system, const js_doc *d, int msgs,
                    int **toks, int *n, char *err, size_t errlen);
/* Token ids of a completion request's `prompt` (node p): a string (encoded
 * with BOS if the tokenizer adds one), or token ids below vocab, alone or as
 * a one-element array. */
int srv_text_tokens(ml_tokenizer *tok, int vocab, const js_doc *d, int p, int **toks, int *n, char *err, size_t errlen);

#endif
