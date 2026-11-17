/* Browser-local bookmarks for exact model settings, operating points and seed cohorts. */
const favouriteStorageKey='claim-dashboard-favourites-v1';
let favouriteNotice='';
const favouriteKeys=loadFavourites();

function loadFavourites() {
  try {
    const stored=JSON.parse(localStorage.getItem(favouriteStorageKey) || '[]');
    if (!Array.isArray(stored)) throw new Error('Invalid saved favourites.');
    return new Set(stored.filter(key=>typeof key==='string'));
  } catch {
    favouriteNotice='Saved favourites could not be read. Favourites are available for this session.';
    return new Set();
  }
}
function saveFavourites() {
  try {
    localStorage.setItem(favouriteStorageKey,JSON.stringify([...favouriteKeys]));
    favouriteNotice='';
  } catch {
    favouriteNotice='This browser could not save favourites. Changes last only for this session.';
  }
  updateFavouriteStatus();
}
function updateFavouriteStatus() {
  elements.favouriteCount.textContent='('+favouriteKeys.size+')';
  elements.favouriteStatus.textContent=favouriteNotice;
  elements.favouriteStatus.hidden=!favouriteNotice;
}
function favouriteButton(row) {
  const button=document.createElement('button'),saved=favouriteKeys.has(row.favourite_key);
  const action=(saved ? 'Remove ' : 'Add ')+row.labels.join(', ')+(saved ? ' from' : ' to')+' favourites';
  button.type='button';button.className='favourite-button';button.dataset.favourite=row.favourite_key;
  button.textContent=saved ? '\u2605' : '\u2606';button.title=action;
  button.setAttribute('aria-label',action);button.setAttribute('aria-pressed',String(saved));
  return button;
}
function toggleFavourite(key) {
  if (favouriteKeys.has(key)) favouriteKeys.delete(key);
  else favouriteKeys.add(key);
  saveFavourites();
  if (elements.favouritesOnly.checked) query();
  else render();
}
function setupFavourites() {
  updateFavouriteStatus();
  elements.favouritesOnly.addEventListener('change',query);
}
