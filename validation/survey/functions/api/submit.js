// Cloudflare Pages Function — requires SURVEY_KV binding
// If deploying elsewhere, swap for your preferred backend
export async function onRequestPost(context) {
  try {
    const data = await context.request.json();
    data._server_ts = new Date().toISOString();
    data._id = crypto.randomUUID();
    const key = `response:${data._id}`;
    await context.env.SURVEY_KV.put(key, JSON.stringify(data));
    const countStr = await context.env.SURVEY_KV.get('response_count') || '0';
    await context.env.SURVEY_KV.put('response_count', String(parseInt(countStr, 10) + 1));
    return new Response(JSON.stringify({ ok: true, id: data._id }), {
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  } catch (e) {
    return new Response(JSON.stringify({ ok: false, error: e.message }), {
      status: 500, headers: { 'Content-Type': 'application/json' },
    });
  }
}
