# vicki_chat

Agente de selección de personal (FastAPI + LangGraph + Qdrant). Lo consume
`ever` en `/vicki` a través de `/api/vicki/*`.

## CVs en el chat (barra lateral)

La respuesta de `POST /chat` trae, además del texto, la lista `candidatos`:
los CVs que entraron al contexto, con `documento_id`, nombre, afinidad y si
hay archivo/miniatura para mostrar. Sale de los **mismos hits** de Qdrant que
arman el contexto (`tools.search_cvs`), así que no hay una segunda búsqueda ni
un segundo embedding; el único costo extra es una consulta a Postgres por
mensaje para resolver el documento de cada candidato.

Esa lista se guarda en `agent.chat_messages.metadata` para poder reconstruir
la barra al recargar (`GET /history/{session_id}` la devuelve).

### Archivo del CV

| ruta | qué devuelve |
|---|---|
| `GET /cv/{documento_id}/thumb` | JPG de la primera página |
| `GET /cv/{documento_id}/file` | PDF (o el original si no hay PDF) |
| `GET /cv/{documento_id}/texto` | texto de la base — contingencia sin archivo |

Los archivos los escribe `vicki_mail` en `CV_STORE_DIR`
(`<hash[:2]>/<hash>/{original.ext,doc.pdf,thumb.jpg}`); acá el volumen se monta
**de solo lectura**. La ruta no está en la base: se deriva de `hash_archivo`,
que ya es UNIQUE en `rag_system.documento_aprobado`. Drive queda como archivo
de respaldo, no se le pega en cada vista.

### Descartes ("tacho")

`agent.chat_descartes (session_id, candidato_id)`. Sacan al candidato de la
conversación: no vuelve a la barra **ni a las búsquedas de esa sesión** — se
excluye con un `must_not` en Qdrant (`tools._filtro_descartes`), no filtrando
después, para que un descartado no ocupe un lugar del `top_k`. Nada se borra
de Postgres ni de Qdrant.

| ruta | |
|---|---|
| `GET /descartes/{session_id}` | listar |
| `POST /descartes/{session_id}` | `{candidato_id}` |
| `DELETE /descartes/{session_id}/{candidato_id}` | deshacer |

El filtro necesita índice de payload en `metadata.candidato_id`: lo crea
`tools.ensure_indices_cv()` al arrancar (idempotente).
