import { chapters, heroReel, liveSession } from './content.mjs?rev=design-18';
import { chapterTabs, initChapter } from './chapters.mjs?rev=design-19';
import { initLive } from './live.mjs?rev=design-15';
import { initReel } from './reel.mjs?rev=design-17';
import { loadShowcase } from './showcase.mjs?rev=showcase-7';

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function updateSelection(group, selectedId) {
  for (const button of group.querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.id === selectedId));
}

function buildSelector(group, items, select, buttonClass) {
  items.forEach((item) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = buttonClass;
    button.dataset.id = item.id;
    if (item.thumb) {
      const thumb = document.createElement('span');
      thumb.className = 'tab-thumb';
      thumb.setAttribute('aria-hidden', 'true');
      thumb.dataset.thumb = item.thumb.src;
      thumb.style.backgroundSize = item.thumb.size;
      thumb.style.backgroundPosition = item.thumb.position;
      button.append(thumb);
    }
    if (item.thumb) button.setAttribute('aria-label', item.label);
    else button.append(document.createTextNode(item.label));
    button.addEventListener('click', () => {
      select(item.id);
      updateSelection(group, item.id);
    });
    group.append(button);
  });
  group.addEventListener('keydown', (event) => {
    const buttons = [...group.querySelectorAll('button')];
    let index = buttons.indexOf(document.activeElement);
    if (index < 0) return;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') index = (index + 1) % buttons.length;
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') index = (index - 1 + buttons.length) % buttons.length;
    else if (event.key === 'Home') index = 0;
    else if (event.key === 'End') index = buttons.length - 1;
    else return;
    event.preventDefault();
    buttons[index].click();
    buttons[index].focus();
  });
  updateSelection(group, items[0].id);
  group.hidden = false;
}

initLive(document.getElementById('live'), liveSession, reducedMotion);

const copyCite = document.querySelector('[data-cite-copy]');
copyCite.addEventListener('click', () => {
  navigator.clipboard.writeText(document.querySelector('[data-cite-bibtex]').textContent)
    .then(() => { copyCite.textContent = 'Copied'; }, () => { copyCite.textContent = 'Select the text to copy'; })
    .then(() => setTimeout(() => { copyCite.textContent = 'Copy BibTeX'; }, 2000));
});

loadShowcase().then((showcase) => {
  initReel(document.querySelector('[data-reel]'), showcase, heroReel, reducedMotion);
  for (const chapter of chapters) {
    const section = document.getElementById(chapter.id);
    const player = initChapter(section, chapter, showcase, reducedMotion);
    if (chapter.clips.length > 1) buildSelector(section.querySelector('[data-chapter-tabs]'), chapterTabs(chapter, showcase), player.select, 'chapter-tab');
  }
}).catch(() => {
  for (const message of document.querySelectorAll('[data-showcase-error]')) message.hidden = false;
});
