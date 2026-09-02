/**
 * IndexedDB persistence for generation history.
 * Images are large base64 strings — localStorage would blow its ~5 MB quota,
 * so history lives in IndexedDB and survives app restarts.
 */

import { GenerationResult } from "../stores/appStore";

const DB_NAME = "aura-studio";
const DB_VERSION = 1;
const STORE = "generations";
export const MAX_HISTORY = 50;

let dbPromise: Promise<IDBDatabase> | null = null;

function openDB(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          const store = db.createObjectStore(STORE, { keyPath: "id" });
          store.createIndex("timestamp", "timestamp");
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

function requestAsPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveGeneration(result: GenerationResult): Promise<void> {
  const db = await openDB();
  const tx = db.transaction(STORE, "readwrite");
  const store = tx.objectStore(STORE);
  await requestAsPromise(store.put(result));

  // Trim to MAX_HISTORY by timestamp (delete oldest overflow).
  const countReq = store.count();
  const count = await requestAsPromise(countReq);
  if (count > MAX_HISTORY) {
    const index = store.index("timestamp");
    const cursorReq = index.openCursor();
    let toDelete = count - MAX_HISTORY;
    cursorReq.onsuccess = () => {
      const cursor = cursorReq.result;
      if (cursor && toDelete > 0) {
        cursor.delete();
        toDelete--;
        cursor.continue();
      }
    };
  }
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function listGenerations(limit: number = MAX_HISTORY): Promise<GenerationResult[]> {
  const db = await openDB();
  const tx = db.transaction(STORE, "readonly");
  const index = tx.objectStore(STORE).index("timestamp");
  // Newest first: iterate the timestamp index backwards.
  const results: GenerationResult[] = [];
  await new Promise<void>((resolve, reject) => {
    const req = index.openCursor(null, "prev");
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor && results.length < limit) {
        results.push(cursor.value as GenerationResult);
        cursor.continue();
      } else {
        resolve();
      }
    };
    req.onerror = () => reject(req.error);
  });
  return results;
}

export async function deleteGeneration(id: string): Promise<void> {
  const db = await openDB();
  await requestAsPromise(db.transaction(STORE, "readwrite").objectStore(STORE).delete(id));
}

export async function clearGenerations(): Promise<void> {
  const db = await openDB();
  await requestAsPromise(db.transaction(STORE, "readwrite").objectStore(STORE).clear());
}
