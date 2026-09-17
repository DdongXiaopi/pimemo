import { resolve } from "node:path";

import { SemanticMemoryStore } from "../../pi-main/pi-main/packages/coding-agent/examples/extensions/semantic-memory/store.ts";

const directory = process.env.PI_SEMANTIC_MEMORY_DIR;
const repository = process.env.PI_AB_WORKDIR;
if (!directory || !repository) throw new Error("PI_SEMANTIC_MEMORY_DIR and PI_AB_WORKDIR are required");

const store = new SemanticMemoryStore(directory);
await store.load();
const scope = { repository: resolve(repository) };
const memories = [
	"For this evaluation project, use node --test tests to run the test suite. Do not use npm test.",
	"The user preference for this evaluation is test-first: verify the test command before suggesting alternatives.",
	"Past experience: npm test fails in this evaluation project; use node --test tests instead.",
];
for (const text of memories) {
	await store.add({ text, category: "fact", scope, source: { sourceType: "explicit" }, importance: 0.9 });
}
console.log(JSON.stringify({ directory: store.filePath, repository: scope.repository, count: memories.length }));
store.close();
