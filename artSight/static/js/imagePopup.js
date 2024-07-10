// imagePopup.js

function openImagePopup(imageSrc, imageTitle) {
    const width = 800;
    const height = 600;
    const left = (screen.width - width) / 2;
    const top = (screen.height - height) / 2;

    const popupWindow = window.open('', '_blank', `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`);

    popupWindow.document.write(`
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <title>${imageTitle}</title>
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #000;
                }
                img {
                    max-width: 100%;
                    max-height: 100%;
                    object-fit: contain;
                }
            </style>
        </head>
        <body>
            <img src="${imageSrc}" alt="${imageTitle}">
        </body>
        </html>
    `);

    popupWindow.document.close();
}