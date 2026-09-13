# Astro Starter Kit: Minimal

```sh
npm create astro@latest -- --template minimal
```

> 🧑‍🚀 **Seasoned astronaut?** Delete this file. Have fun!

## 🚀 Project Structure

Inside of your Astro project, you'll see the following folders and files:

```text
/
├── public/
├── src/
│   └── pages/
│       └── index.astro
└── package.json
```

Astro looks for `.astro` or `.md` files in the `src/pages/` directory. Each page is exposed as a route based on its file name.

There's nothing special about `src/components/`, but that's where we like to put any Astro/React/Vue/Svelte/Preact components.

Any static assets, like images, can be placed in the `public/` directory.

## 🧞 Commands

All commands are run from the root of the project, from a terminal:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Installs dependencies                            |
| `npm run dev`             | Starts local dev server at `localhost:4321`      |
| `npm run build`           | Build your production site to `./dist/`          |
| `npm run preview`         | Preview your build locally, before deploying     |
| `npm run astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `npm run astro -- --help` | Get help using the Astro CLI                     |

## 👀 Want to learn more?

Feel free to check [our documentation](https://docs.astro.build) or jump into our [Discord server](https://astro.build/chat).

## Design

Písma a neutrály přebírají design systém Prvni pozice (`web-1P`):
Roboto + Roboto Mono (self-hostované přes @fontsource, žádný požadavek
na Google Fonts), neutrální škála `--n-*`, značková limetka `#98c800`.

Limetka se používá **decentně** — odkazy, nadpisek v hlavičce, linka nad
odbornou sekcí. Není to marketingová stránka, je to výzkumný deník.

Barevné značky výsledků (zelená / červená / oranžová / fialová) jsou
vlastní a nesou informaci, ne dekoraci: vyšlo / nevyšlo / napůl / zásadní
nález. Proto se nesjednocují se značkovou paletou.

## Videa a upoutávky

Videa leží v `public/videa/`, odkazují se z `src/data/pokusy.json` klíčem
`video` (`soubor`, `popis` pro čtečky, `titulek` pod video v článku).

Pás na domovské stránce bere tři id z `<Upoutavky ids={[...]} />`.
Karta bez videa se nepřeskakuje — ukáže šrafovanou plochu se štítkem
„brzy" a zůstává plnohodnotným odkazem na článek. Doplnit video znamená
jen nahrát soubor a přidat klíč, nikde jinde se nesahá.

Na úzké obrazovce se karty skládají pod sebe (`auto-fit, minmax(220px, 1fr)`).
Videa v pásu jsou bez ovládání, ztlumená a ve smyčce; v článku mají ovládání.
